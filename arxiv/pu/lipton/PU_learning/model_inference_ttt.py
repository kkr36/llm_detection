"""
Test-time-training inference for the xy/xz evaluation setting.

get_preds_xy_ttt mirrors model_inference.get_preds_xy exactly, except the unlabeled
(U) predictions come from algorithm_ttt.ttt_adapt_and_predict instead of the frozen
estimator.u_probs. The positive (P) reference set is still scored with a frozen
forward (p_probs). This lets eval_TEDn_X_rewrite_Z_ttt.py compare ttt_mode in
{none, episodic, online} on the same rewrite_Z stream.
"""

import os
import numpy as np
import pandas as pd
import torch

# Reuse the exact data-construction helpers from model_inference (names are bound in
# that module via its `from helper/data_helper import *`).
from model_inference import (
    _xy_eval_parquet,
    split_into_sentences,
    clean_text,
    initialize_bert_transform,
    IMDbBERTData,
    PosData,
    p_probs,
    get_u_data_xy,
)

from algorithm_ttt import ttt_adapt_and_predict, ttt_stream_predict, ttt_batch_stream_predict


class _OrderSampler(torch.utils.data.Sampler):
    """Yield dataset indices in a fixed, caller-supplied order."""
    def __init__(self, order):
        self.order = list(order)
    def __iter__(self):
        return iter(self.order)
    def __len__(self):
        return len(self.order)


def _interleave_order(true_targets):
    """Interleave human (true_target==0) and AI (true_target==1) positions:
    h0, a0, h1, a1, ...  (leftover of the longer class appended at the end).
    Under test_alpha=0.5 the two classes are equal in size => perfect alternation."""
    tt = np.asarray(true_targets)
    human = np.where(tt == 0)[0]
    ai = np.where(tt == 1)[0]
    order = []
    for h, a in zip(human, ai):
        order.append(int(h))
        order.append(int(a))
    n = min(len(human), len(ai))
    order.extend(int(i) for i in (human[n:] if len(human) > len(ai) else ai[n:]))
    return order


def get_preds_xy_ttt(net, device, test_alpha, flip, seed, sentence, clean, llm_col, ttt_cfg):
    """Same contract as model_inference.get_preds_xy, but the U-set is adapted at
    test time per ttt_cfg (a dict of ttt_adapt_and_predict kwargs, incl. `mode`)."""
    data_path = _xy_eval_parquet()
    arxiv_data = pd.read_parquet(data_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    cal_data = arxiv_data.iloc[-2000:].reset_index(drop=True)

    # Positive set: first 500 rows, human_abstract only (frozen forward)
    pos_texts = cal_data.iloc[:500]["human_abstract"].tolist()
    if sentence:
        pos_texts, _ = split_into_sentences(pos_texts, [1] * len(pos_texts))
    if clean:
        pos_texts = clean_text(pos_texts)

    transform = initialize_bert_transform('distilbert-base-uncased')
    pos_dataset = IMDbBERTData(pos_texts, [1] * len(pos_texts), transform=transform)
    p_data = PosData(transform=pos_dataset.transform,
                     target_transform=pos_dataset.target_transform,
                     data=pos_dataset.p_data,
                     index=np.array(range(len(pos_dataset.p_data))), data_type="xy")
    p_loader = torch.utils.data.DataLoader(p_data, batch_size=16, shuffle=False)

    # Unlabeled stream (shuffle=False -> deterministic order for online TTT)
    u_loader, _, _ = get_u_data_xy(test_alpha, flip, seed, sentence, clean, llm_col)

    # P-set frozen (scored before any adaptation mutates the net)
    pos_probs = p_probs(net, device, p_loader)
    # U-set adapt-then-predict
    unlabeled_probs, unlabeled_targets = ttt_adapt_and_predict(net, device, u_loader, **ttt_cfg)

    if not flip:
        pos_probs = 1 - pos_probs
        unlabeled_probs = 1 - unlabeled_probs

    return pos_probs, unlabeled_probs, unlabeled_targets


def _frac_z(f, growth):
    """Fraction of AI slots that are rewrite_Z at stream fraction f in [0,1].
    growth=0 -> all rewrite_X (train-distribution AI); growth=1 -> all rewrite_Z (abrupt
    shift, the prior behavior); 0<growth<1 -> linear ramp reaching 100% Z at stream
    fraction (1-growth)/(2*growth)... calibrated so growth=0.5 hits 100% Z at f=0.5
    (halfway batch), then stays 100%."""
    if growth <= 0:
        return 0.0
    if growth >= 1:
        return 1.0
    slope = 2.0 * growth / (1.0 - growth)      # growth=0.5 -> slope 2 -> 100% Z at f=0.5
    return min(1.0, max(0.0, slope * f))


class _OrderedStream(torch.utils.data.Dataset):
    """Pre-tokenized stream in a FIXED order; yields (index, packed(seq,2), 1, true_target)
    to match the (index, inputs, target, true_target) loader contract."""
    def __init__(self, packed, true_targets):
        self.packed = packed
        self.tt = np.asarray(true_targets)

    def __len__(self):
        return len(self.tt)

    def __getitem__(self, i):
        return i, torch.as_tensor(self.packed[i], dtype=torch.float), 1, int(self.tt[i])


def _build_gradual_order(seed, sentence, clean, growth, batch_size):
    """Gradual-shift U-stream: interleaved human/AI batches (16 human + 16 AI per 32-batch),
    where the AI half of batch b draws rewrite_Z with probability _frac_z(b/B, growth) and
    rewrite_X otherwise. Same source rows as get_u_data_xy (cal split rows 500:), so it's
    comparable to the non-gradual runs. Returns (texts, true_targets{0=human,1=ai},
    ai_source{'human','X','Z'}, n_batches)."""
    data_path = _xy_eval_parquet()
    arxiv = pd.read_parquet(data_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    # Held-out block = last 2000 rows (disjoint from this seed's train rows [0:8000]).
    # Default: reserve its first 500 rows for the frozen P-set calibration reference
    # (get_preds_xy_ttt_stream reads cal_data.iloc[:500]) and stream only the rest (1500).
    # TTT_FULL_HELDOUT=1 reclaims those 500 into the stream (~+33% batches). The P-set still
    # reads those same held-out humans, so it stays unseen-by-training; the only effect is a
    # small P-set/stream human overlap that touches just the FPR-5% threshold + BBE, not the
    # AUC / accuracy / recall / P(AI|AI) / P(human|human) curves. Rows stay mutually exclusive
    # from training either way.
    u_block = arxiv.iloc[-2000:].reset_index(drop=True)
    u = u_block if os.environ.get("TTT_FULL_HELDOUT", "0") == "1" else u_block.iloc[500:]
    human = u["human_abstract"].tolist()
    ax = u["rewrite_X"].tolist()
    az = u[os.environ.get("TTT_EVAL_COL", "rewrite_Z")].tolist()   # gradual target column (X -> this)
    if sentence:
        human, _ = split_into_sentences(human, [0] * len(human))
        ax, _ = split_into_sentences(ax, [0] * len(ax))
        az, _ = split_into_sentences(az, [0] * len(az))
    if clean:
        human, ax, az = clean_text(human), clean_text(ax), clean_text(az)
    rng = np.random.default_rng(seed)
    for pool in (human, ax, az):
        rng.shuffle(pool)

    half = batch_size // 2                    # per batch: `half` human + `half` AI
    if half == 0:
        raise ValueError(
            f"batch_size={batch_size} is too small for _build_gradual_order: it needs "
            "batch_size // 2 >= 1 so each batch can hold at least one human + one AI example."
        )
    n_ai = min(len(human), len(ax), len(az))
    n_batches = n_ai // half
    texts, tt, src = [], [], []
    hi = xi = zi = 0
    # Running-remainder (Bresenham-style) allocator: _frac_z is a smooth ramp, but each batch
    # can only hold an integer count of Z slots. Naively rounding half*pz per batch collapses
    # the ramp into 1-3 hard steps when `half` is small (e.g. half=1 is a single-batch flip from
    # 0% to 100% adversarial). Instead we accumulate the smooth target and only emit a Z slot
    # once the cumulative target crosses the next integer, so the total Z count still matches
    # the _frac_z integral but individual batches near the transition are a mix of X/Z.
    z_target_cum = 0.0
    z_allocated = 0
    for b in range(n_batches):
        pz = _frac_z(b / n_batches, growth)
        z_target_cum += half * pz
        n_z = max(0, min(half, int(round(z_target_cum)) - z_allocated))
        z_allocated += n_z
        n_x = half - n_z
        ai_items = ([(az[zi + k], "Z") for k in range(n_z)]
                    + [(ax[xi + k], "X") for k in range(n_x)])
        zi += n_z
        xi += n_x
        for k in range(half):                 # interleave human, AI within the batch
            texts.append(human[hi + k]); tt.append(0); src.append("human")
            t, s = ai_items[k]; texts.append(t); tt.append(1); src.append(s)
        hi += half
    return texts, np.array(tt), src, n_batches


def get_preds_xy_ttt_stream(net, device, test_alpha, flip, seed, sentence, clean,
                            llm_col, ttt_cfg, order="interleave", growth=None):
    """Streaming variant used for continual-learning curves. Identical P/U data
    construction to model_inference.get_preds_xy (the exact eval used by
    prepare_optimization.py): same _xy_eval_parquet, last-2000-row cal split, P-set =
    first 500 human_abstract, U-set = rows 500: of human_abstract + `llm_col`, sampled
    to test_alpha via get_u_data_xy. Here the U-set is scored one sample at a time
    (ttt_stream_predict) in a human/AI-alternating order (order='interleave').

    Two polarity conventions are kept separate:
      - per-sample accuracy uses the NATIVE model output P(AI)=softmax[:,0] (so the
        `correct` column is right regardless of the flip bookkeeping);
      - the returned metric arrays follow prepare_optimization's `flip` convention
        (flip=False inverts, so auc_fn treats human as positive) so AUC matches.

    Returns (pos_probs, unlabeled_probs, unlabeled_targets, records_df) in stream order.
    """
    data_path = _xy_eval_parquet()
    arxiv_data = pd.read_parquet(data_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    cal_data = arxiv_data.iloc[-2000:].reset_index(drop=True)

    pos_texts = cal_data.iloc[:500]["human_abstract"].tolist()
    if sentence:
        pos_texts, _ = split_into_sentences(pos_texts, [1] * len(pos_texts))
    if clean:
        pos_texts = clean_text(pos_texts)

    transform = initialize_bert_transform('distilbert-base-uncased')
    pos_dataset = IMDbBERTData(pos_texts, [1] * len(pos_texts), transform=transform)
    p_data = PosData(transform=pos_dataset.transform,
                     target_transform=pos_dataset.target_transform,
                     data=pos_dataset.p_data,
                     index=np.array(range(len(pos_dataset.p_data))), data_type="xy")
    p_loader = torch.utils.data.DataLoader(p_data, batch_size=16, shuffle=False)

    bs = int(ttt_cfg.get("batch_size", 32))
    if growth is not None:
        # Gradual-shift stream: AI half of each batch ramps rewrite_X -> rewrite_Z.
        g_texts, g_tt, g_src, _nb = _build_gradual_order(seed, sentence, clean, growth, bs)
        packed = transform(g_texts)
        stream_loader = torch.utils.data.DataLoader(
            _OrderedStream(packed, g_tt), batch_size=bs, shuffle=False)
        stream_texts, stream_src = g_texts, g_src
    else:
        # Same U construction as prepare_optimization (get_u_data_xy). u_texts positions
        # align with the dataset index; true_target 0=human, 1=AI. flip doesn't change the
        # data (only later probability polarity).
        u_loader, u_texts, u_labels = get_u_data_xy(test_alpha, flip, seed, sentence, clean, llm_col)
        dataset = u_loader.dataset
        if order == "interleave":
            idx_order = _interleave_order(dataset.true_targets)
            stream_loader = torch.utils.data.DataLoader(
                dataset, batch_size=bs, sampler=_OrderSampler(idx_order))
        else:
            stream_loader = u_loader
        stream_texts, stream_src = u_texts, None

    pos_probs = p_probs(net, device, p_loader)                     # frozen reference
    pset_pai_native = np.asarray(pos_probs).copy()                 # native P(AI) on human P-set (for threshold calibration)
    # adaptation granularity: 'sample' (per-instance) or 'batch' (adapt on the whole
    # loader batch per MLM step -- cleaner gradient). Default 'sample' for back-compat.
    cfg = dict(ttt_cfg)
    adapt_granularity = cfg.pop("adapt_granularity", "sample")
    if adapt_granularity == "batch":
        out = ttt_batch_stream_predict(net, device, stream_loader, **cfg)
    else:
        out = ttt_stream_predict(net, device, stream_loader, **cfg)

    probs_native = out["probs"]                 # softmax[:,0] = P(AI), native polarity
    targets = out["targets"]                    # 0 = human, 1 = AI

    # ---- per-sample accuracy from native P(AI) ----
    prob_ai = probs_native[:, 0]
    is_ai = (targets == 1).astype(int)
    pred_ai = (prob_ai > 0.5).astype(int)
    correct = (pred_ai == is_ai).astype(int)

    text_index = out["text_index"]
    texts = [stream_texts[i] for i in text_index]
    records_df = pd.DataFrame({
        "sample_id": out["sample_ids"],
        "batch_id": out["batch_ids"],
        "text_index": text_index,
        "text": texts,
        "label": np.where(is_ai == 1, "ai", "human"),
        "target": targets,
        "prob_ai": prob_ai,
        "pred_ai": pred_ai,
        "correct": correct,
    })
    if stream_src is not None:                 # gradual regime: record X vs Z per AI item
        records_df["ai_source"] = [stream_src[i] for i in text_index]

    # ---- aggregate-metric arrays follow prepare_optimization's flip convention ----
    unlabeled_probs = probs_native.copy()
    if not flip:
        pos_probs = 1 - pos_probs
        unlabeled_probs = 1 - unlabeled_probs

    return pos_probs, unlabeled_probs, targets, records_df, pset_pai_native

