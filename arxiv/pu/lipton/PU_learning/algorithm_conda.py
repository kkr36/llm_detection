"""ConDA training: NT-Xent contrastive loss, MMD domain alignment, on-the-fly
token-dropout augmentation, and the combined training loop.

Loss (per batch), following Bhattacharjee et al. (IJCNLP 2023):
    loss = (1 - lambda_w) * CE_source
         + lambda_w       * (NTXent_source + NTXent_target) / 2
         + lambda_mmd     * MMD(emb_source, emb_target)
where CE_source is averaged over the clean and perturbed source views.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from tqdm import tqdm

PAD_ID = 0  # distilbert-base-uncased pad_token_id


def token_dropout(x, p=0.2, generator=None):
    """Produce a perturbed view of a tokenized batch by randomly dropping tokens.

    x: LongTensor [batch, seq, 2] with x[:,:,0]=input_ids, x[:,:,1]=attention_mask.
    Randomly (prob p) zeroes attention and pads input_ids for real, non-[CLS] tokens.
    Cheap tensor op on already-tokenized inputs (no re-tokenization).
    """
    x = x.clone()
    input_ids = x[:, :, 0]
    attn = x[:, :, 1]

    real = attn.bool().clone()
    real[:, 0] = False  # never drop the leading [CLS] token

    rand = torch.rand(input_ids.shape, device=x.device, generator=generator)
    drop = real & (rand < p)

    input_ids[drop] = PAD_ID
    attn[drop] = 0
    x[:, :, 0] = input_ids
    x[:, :, 1] = attn
    return x


def nt_xent(z1, z2, temperature=0.5):
    """SimCLR NT-Xent loss between two sets of projected embeddings.

    z1, z2: [batch, dim] projections of two augmented views. Positive pair for
    row i is (z1[i], z2[i]); all other 2N-2 samples are negatives.
    """
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)               # [2N, d]
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / temperature   # [2N, 2N]

    # mask out self-similarity
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float("-inf"))

    # positive index: i <-> i+n
    pos_idx = torch.arange(2 * n, device=z.device)
    pos_idx = (pos_idx + n) % (2 * n)
    return F.cross_entropy(sim, pos_idx)


def mmd(x, y, kernel="rbf"):
    """Empirical maximum mean discrepancy between feature batches x, y ([n, d]).

    Multi-bandwidth kernel, ported from the ConDA repo's mmd_code.py.
    """
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * xy

    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(xy)

    if kernel == "multiscale":
        bandwidths = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidths:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1
    else:  # rbf / gaussian
        bandwidths = [10, 15, 20, 50]
        for a in bandwidths:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


def train_conda(epoch, net, source_loader, target_loader, optimizer, device,
                lambda_w=0.5, lambda_mmd=1.0, temperature=0.5, dropout_p=0.2,
                class_weight=None, show_bar=True):
    """One ConDA training epoch. Returns (source_acc, avg_ce, avg_ctr, avg_mmd).

    class_weight: optional [num_classes] tensor of per-class CE weights (e.g.
    inverse class frequency) to counteract a skewed labeled source.
    """
    net.train()
    ce_criterion = nn.CrossEntropyLoss(weight=class_weight)

    total = 0
    correct = 0
    ce_sum = ctr_sum = mmd_sum = 0.0
    n_batches = 0

    # target is unlabeled and usually smaller/larger; cycle it against source.
    loader = zip(source_loader, cycle(target_loader))
    n_steps = len(source_loader)
    iterator = tqdm(loader, total=n_steps, desc=f"ConDA epoch {epoch}", disable=not show_bar)

    for src_batch, tgt_batch in iterator:
        src_x, src_y = src_batch
        tgt_x = tgt_batch[0]

        src_x = src_x.to(device)
        src_y = src_y.to(device)
        tgt_x = tgt_x.to(device)

        src_x_p = token_dropout(src_x, p=dropout_p)
        tgt_x_p = token_dropout(tgt_x, p=dropout_p)

        optimizer.zero_grad()

        emb_s, proj_s, logits_s = net(src_x, return_features=True)
        _, proj_sp, logits_sp = net(src_x_p, return_features=True)
        emb_t, proj_t, _ = net(tgt_x, return_features=True)
        _, proj_tp, _ = net(tgt_x_p, return_features=True)

        ce = (ce_criterion(logits_s, src_y) + ce_criterion(logits_sp, src_y)) / 2.0
        ctr = (nt_xent(proj_s, proj_sp, temperature)
               + nt_xent(proj_t, proj_tp, temperature)) / 2.0
        mmd_loss = mmd(emb_s, emb_t)

        loss = (1 - lambda_w) * ce + lambda_w * ctr + lambda_mmd * mmd_loss
        loss.backward()
        optimizer.step()

        ce_sum += ce.item()
        ctr_sum += ctr.item()
        mmd_sum += mmd_loss.item()
        n_batches += 1

        _, predicted = logits_s.max(1)
        total += src_y.size(0)
        correct += predicted.eq(src_y).sum().item()

        if show_bar:
            iterator.set_postfix(
                acc=f"{100.*correct/max(total,1):.1f}",
                ce=f"{ce_sum/n_batches:.3f}",
                ctr=f"{ctr_sum/n_batches:.3f}",
                mmd=f"{mmd_sum/n_batches:.4f}",
            )

    n_batches = max(n_batches, 1)
    return (100. * correct / max(total, 1),
            ce_sum / n_batches, ctr_sum / n_batches, mmd_sum / n_batches)
