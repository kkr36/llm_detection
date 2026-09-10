"""
Test-Time Training (Sun et al., 2020) for the DistilBERT AI-detector.

- train_PN_ttt: fully-supervised PN training with an auxiliary MLM self-supervision
  loss on the shared trunk (joint training -- the config that makes test-time SSL
  updates transfer to the main task). Drop-in replacement for algorithm.train_PN;
  same data, labels, optimizer, and classification objective, plus one MLM term.

- ttt_adapt_and_predict: the test-time engine. For each unlabeled batch it takes a
  few gradient steps on the MLM loss (updating the trunk / LayerNorm only), then
  classifies with the adapted features. Returns (probs, targets) shaped exactly like
  estimator.u_probs so downstream metrics are untouched.
    * mode='none'     -> no adaptation (frozen baseline).
    * mode='episodic' -> reset trunk to trained weights before each batch.
    * mode='online'   -> accumulate adaptation across the stream (continual).
"""

import numpy as np
import torch
import torch.nn as nn

from utils import progress_bar
from models.bert_ttt import mask_tokens, pack_ids


def _base(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def train_PN_ttt(epoch, net, u_trainloader, optimizer, criterion, device,
                 mlm_weight=1.0, mlm_prob=0.15, show_bar=True):
    """PN training + auxiliary MLM loss. Mirrors algorithm.train_PN."""
    if show_bar:
        print('\nTrain (PN+TTT) Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    cls_loss_sum = 0
    mlm_loss_sum = 0
    correct = 0
    total = 0

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for batch_idx, (_, inputs, _, targets) in enumerate(u_trainloader):
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)

        # main (classification) loss -- identical to train_PN
        outputs = net(inputs)
        cls_loss = criterion(outputs, targets)

        # auxiliary self-supervised MLM loss on the same batch
        input_ids = inputs[:, :, 0]
        attn = inputs[:, :, 1]
        masked_ids, mlm_labels = mask_tokens(input_ids, attn, mlm_prob=mlm_prob)
        masked_packed = pack_ids(masked_ids, attn)
        mlm_logits = net(masked_packed, task="mlm")
        mlm_loss = mlm_criterion(
            mlm_logits.reshape(-1, mlm_logits.size(-1)), mlm_labels.reshape(-1))

        loss = cls_loss + mlm_weight * mlm_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        cls_loss_sum += cls_loss.item()
        mlm_loss_sum += mlm_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += np.sum(predicted.eq(targets).cpu().numpy())

        if show_bar:
            progress_bar(batch_idx, len(u_trainloader),
                         'Loss: %.3f (cls %.3f | mlm %.3f) | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), cls_loss_sum / (batch_idx + 1),
                            mlm_loss_sum / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total


def _make_inner_optimizer(params, inner_lr, optimizer_type):
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=inner_lr)
    elif optimizer_type == "adam":
        return torch.optim.Adam(params, lr=inner_lr)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=inner_lr)
    raise ValueError(f"Unknown inner optimizer: {optimizer_type}")


def ttt_adapt_and_predict(net, device, u_loader, mode="episodic", inner_lr=1e-3,
                          n_steps=1, adapt_scope="trunk", mlm_prob=0.15,
                          n_mask_samples=1, optimizer_type="sgd", show_bar=True):
    """Adapt on the MLM loss per batch, then classify. Returns (probs (N,2), targets (N,))."""
    base = _base(net)
    base.to(device)

    # frozen baseline: just a forward pass (equivalent to estimator.u_probs)
    if mode == "none":
        base.eval()
        return _frozen_u_probs(base, device, u_loader, show_bar=show_bar)

    # snapshot trained weights for reset / final restore
    snapshot = {k: v.detach().clone() for k, v in base.state_dict().items()}

    adapt_params = base.adapt_params(adapt_scope)
    # only compute grads for the params we actually update
    for p in base.parameters():
        p.requires_grad_(False)
    for p in adapt_params:
        p.requires_grad_(True)

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    online_opt = None
    if mode == "online":
        online_opt = _make_inner_optimizer(adapt_params, inner_lr, optimizer_type)

    all_probs, all_targets = [], []

    for batch_idx, (_, inputs, _, targets) in enumerate(u_loader):
        inputs = inputs.to(device)

        if mode == "episodic":
            base.load_state_dict(snapshot)  # reset trunk+heads to trained weights
            opt = _make_inner_optimizer(adapt_params, inner_lr, optimizer_type)
        elif mode == "online":
            opt = online_opt
        else:
            raise ValueError(f"Unknown ttt mode: {mode}")

        # --- adaptation: MLM gradient steps (dropout off for stability) ---
        base.eval()
        input_ids = inputs[:, :, 0]
        attn = inputs[:, :, 1]
        for _step in range(n_steps):
            opt.zero_grad()
            loss = 0.0
            for _ in range(n_mask_samples):
                masked_ids, mlm_labels = mask_tokens(input_ids, attn, mlm_prob=mlm_prob)
                masked_packed = pack_ids(masked_ids, attn)
                logits = base(masked_packed, task="mlm")
                loss = loss + mlm_criterion(
                    logits.reshape(-1, logits.size(-1)), mlm_labels.reshape(-1))
            loss = loss / n_mask_samples
            loss.backward()
            opt.step()

        # --- prediction with adapted features (classifier head unchanged) ---
        with torch.no_grad():
            out = base(inputs)
            probs = torch.nn.functional.softmax(out, dim=-1)
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(targets.numpy())

        if show_bar:
            progress_bar(batch_idx, len(u_loader), f'TTT[{mode}] adapt+predict')

    # restore trained weights and grad flags for a clean net
    base.load_state_dict(snapshot)
    for p in base.parameters():
        p.requires_grad_(True)

    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def ttt_stream_predict(net, device, u_loader, mode="online", batch_size=32,
                       inner_lr=1e-3, n_steps=1, adapt_scope="trunk", mlm_prob=0.15,
                       n_mask_samples=1, optimizer_type="sgd", show_bar=True):
    """Per-sample streaming TTT for continual-learning curves.

    Processes the loader ONE SAMPLE AT A TIME (adaptation granularity = single
    sample), predicting each in stream order. `batch_size` defines the grouping used
    for (a) the batch-level curve and (b) the episodic reset period.
      - mode='none'     : no adaptation (flat baseline).
      - mode='online'   : adapt cumulatively on every sample, never reset (rises,
                          plateaus).
      - mode='episodic' : reset to trained weights at each batch boundary (every
                          `batch_size` samples) and adapt cumulatively *within* the
                          batch (sawtooth: climbs across a batch, drops on reset).

    Returns dict with stream-ordered arrays (all length N):
      probs (N,2), targets (N,), batch_ids (N,), sample_ids (N,), text_index (N,).
    Adaptation uses only the unlabeled text (MLM) -- labels are never used here.
    """
    base = _base(net)
    base.to(device)

    snapshot = {k: v.detach().clone() for k, v in base.state_dict().items()}
    adapt_params = base.adapt_params(adapt_scope)
    for p in base.parameters():
        p.requires_grad_(False)
    for p in adapt_params:
        p.requires_grad_(True)

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    adapting = mode in ("online", "episodic")
    opt = _make_inner_optimizer(adapt_params, inner_lr, optimizer_type) if adapting else None

    probs_all, targ_all, batch_all, sample_all, index_all = [], [], [], [], []
    gi = 0  # global sample counter (stream position)

    for batch_idx, (indices, inputs, _, true_targets) in enumerate(u_loader):
        inputs = inputs.to(device)
        bsz = inputs.size(0)
        for j in range(bsz):
            batch_id = gi // batch_size

            if mode == "episodic" and (gi % batch_size == 0):
                base.load_state_dict(snapshot)  # reset at batch boundary
                opt = _make_inner_optimizer(adapt_params, inner_lr, optimizer_type)

            xi = inputs[j:j + 1]  # single sample, keep batch dim

            if adapting:
                base.eval()  # dropout off for stable adaptation
                ids = xi[:, :, 0]
                attn = xi[:, :, 1]
                for _step in range(n_steps):
                    opt.zero_grad()
                    loss = 0.0
                    for _ in range(n_mask_samples):
                        m_ids, m_lab = mask_tokens(ids, attn, mlm_prob=mlm_prob)
                        logits = base(pack_ids(m_ids, attn), task="mlm")
                        loss = loss + mlm_criterion(
                            logits.reshape(-1, logits.size(-1)), m_lab.reshape(-1))
                    (loss / n_mask_samples).backward()
                    opt.step()

            with torch.no_grad():
                out = base(xi)
                p = torch.nn.functional.softmax(out, dim=-1)

            probs_all.append(p.detach().cpu().numpy().reshape(1, -1))
            targ_all.append(int(true_targets[j]))
            batch_all.append(batch_id)
            sample_all.append(gi)
            index_all.append(int(indices[j]))
            gi += 1

        if show_bar:
            progress_bar(batch_idx, len(u_loader), f'TTT-stream[{mode}]')

    # restore clean net
    base.load_state_dict(snapshot)
    for p in base.parameters():
        p.requires_grad_(True)

    return {
        "probs": np.concatenate(probs_all, axis=0),
        "targets": np.array(targ_all),
        "batch_ids": np.array(batch_all),
        "sample_ids": np.array(sample_all),
        "text_index": np.array(index_all),
    }


def _frozen_batch(base, device, u_loader, show_bar=True):
    probs_all, targ_all, batch_all, sample_all, index_all = [], [], [], [], []
    gi = 0
    with torch.no_grad():
        for bidx, (indices, inputs, _, true_targets) in enumerate(u_loader):
            inputs = inputs.to(device)
            probs = torch.nn.functional.softmax(base(inputs), dim=-1)
            bsz = inputs.size(0)
            probs_all.append(probs.detach().cpu().numpy())
            targ_all.extend(int(t) for t in true_targets)
            batch_all.extend([bidx] * bsz)
            sample_all.extend(range(gi, gi + bsz))
            index_all.extend(int(x) for x in indices)
            gi += bsz
            if show_bar:
                progress_bar(bidx, len(u_loader), 'TTT-batch[none]')
    return {"probs": np.concatenate(probs_all, axis=0), "targets": np.array(targ_all),
            "batch_ids": np.array(batch_all), "sample_ids": np.array(sample_all),
            "text_index": np.array(index_all)}


def ttt_batch_stream_predict(net, device, u_loader, mode="online", batch_size=32,
                             inner_lr=1e-3, n_steps=3, adapt_scope="trunk", mlm_prob=0.15,
                             n_mask_samples=1, optimizer_type="sgd", show_bar=True):
    """Batch-level TTT: adapt on a WHOLE batch of test samples per MLM gradient step
    (much cleaner gradient than per-sample), then predict that batch. The loader yields
    the adaptation batches, so one loader batch == one batch_id.
      - none:     no adaptation (frozen predict).
      - online:   adapt cumulatively across batches, never reset.
      - episodic: reset to trained weights before each batch.
    Returns the same dict shape as ttt_stream_predict. `batch_size` is informational
    here (the loader already batches); batch_id is the loader-batch index."""
    base = _base(net)
    base.to(device)
    if mode == "none":
        base.eval()
        return _frozen_batch(base, device, u_loader, show_bar=show_bar)

    snapshot = {k: v.detach().clone() for k, v in base.state_dict().items()}
    adapt_params = base.adapt_params(adapt_scope)
    for p in base.parameters():
        p.requires_grad_(False)
    for p in adapt_params:
        p.requires_grad_(True)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    opt = _make_inner_optimizer(adapt_params, inner_lr, optimizer_type) if mode == "online" else None

    probs_all, targ_all, batch_all, sample_all, index_all = [], [], [], [], []
    gi = 0
    for bidx, (indices, inputs, _, true_targets) in enumerate(u_loader):
        inputs = inputs.to(device)
        if mode == "episodic":
            base.load_state_dict(snapshot)
            opt = _make_inner_optimizer(adapt_params, inner_lr, optimizer_type)

        base.eval()  # dropout off for stable adaptation
        ids = inputs[:, :, 0]
        attn = inputs[:, :, 1]
        for _step in range(n_steps):
            opt.zero_grad()
            loss = 0.0
            for _ in range(n_mask_samples):
                m_ids, m_lab = mask_tokens(ids, attn, mlm_prob=mlm_prob)
                logits = base(pack_ids(m_ids, attn), task="mlm")
                loss = loss + mlm_criterion(logits.reshape(-1, logits.size(-1)), m_lab.reshape(-1))
            (loss / n_mask_samples).backward()
            opt.step()

        with torch.no_grad():
            probs = torch.nn.functional.softmax(base(inputs), dim=-1)
        bsz = inputs.size(0)
        probs_all.append(probs.detach().cpu().numpy())
        targ_all.extend(int(t) for t in true_targets)
        batch_all.extend([bidx] * bsz)
        sample_all.extend(range(gi, gi + bsz))
        index_all.extend(int(x) for x in indices)
        gi += bsz
        if show_bar:
            progress_bar(bidx, len(u_loader), f'TTT-batch[{mode}]')

    base.load_state_dict(snapshot)
    for p in base.parameters():
        p.requires_grad_(True)
    return {"probs": np.concatenate(probs_all, axis=0), "targets": np.array(targ_all),
            "batch_ids": np.array(batch_all), "sample_ids": np.array(sample_all),
            "text_index": np.array(index_all)}


def _frozen_u_probs(base, device, u_loader, show_bar=True):
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch_idx, (_, inputs, _, targets) in enumerate(u_loader):
            inputs = inputs.to(device)
            out = base(inputs)
            probs = torch.nn.functional.softmax(out, dim=-1)
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(targets.numpy())
            if show_bar:
                progress_bar(batch_idx, len(u_loader), 'TTT[none] frozen predict')
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)
