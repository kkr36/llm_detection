"""Evaluate every trained ConDA model on the LLMs it did NOT see during training
(neither source LLM1 nor target LLM2) -- i.e. out-of-distribution generalization.

With 5 LLMs {GPT OSS 120b, Gemini 3 Preview, Llama 3.3 70b Instruct, Qwen, Codex},
each model saw 2, so it is evaluated on the 3 unseen ones, using the SAME per-LLM
test sets as prepare_heatmap_conda*.py (get_preds_llm, alpha=0.5, human-positive).

Eval always goes through the codex parquet (codex=True): its first 10k rows are
byte-identical to the base parquet, so it can serve every LLM's test set uniformly.

Parallelized as one (pair x unseen-LLM) cell per array task to avoid CSV races:
    python prepare_heatmap_conda_unseen.py count            -> prints number of cells
    python prepare_heatmap_conda_unseen.py cell <idx>       -> compute one cell, write its row
    python prepare_heatmap_conda_unseen.py merge            -> concat all rows -> final CSV
"""
import os
import re
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

from prepare_metrics import *
from estimator import BBE_estimator

# ---- config ----
SWEEPS = [
    ("base",  "/share/garg/arxiv_kaggle/ConDA_llm"),
    ("codex", "/share/garg/arxiv_kaggle/ConDA_llm_codex"),
]
ALL_LLMS = ["GPT_OSS_120b", "Gemini_3_Preview", "Llama_3.3_70b_Instruct", "Qwen", "Codex"]
CELL_DIR = "/share/garg/arxiv_kaggle/ConDA_unseen_cells"
OUTPUT_CSV = "logging_accuracy_llm_conda_unseen.csv"
EPOCHS = 3
TEST_ALPHA = 0.5
TRAIN_ALPHA = 0.5
TEST_YEAR = 2020
TRAIN_YEAR = 2020
TEST_CIS = [.9, .95, .99]
N_BOOTSTRAP = 2500
DATA_TYPE = "ArXiv_BERT"

_pair_re = re.compile(r'^llm_type_(.+)\|(.+)_(\d+)$')


def enumerate_cells():
    """Deterministic list of (sweep, llm1, llm2, held_out_llm, [seeds]).
    One entry per (trained pair) x (LLM unseen in that pair)."""
    cells = []
    for sweep, base in SWEEPS:
        pairs_seeds = defaultdict(list)
        for d in sorted(os.listdir(base)):
            m = _pair_re.match(d)
            if m:
                pairs_seeds[(m.group(1), m.group(2))].append(int(m.group(3)))
        for (l1, l2) in sorted(pairs_seeds):
            seen = {l1, l2}
            for held in ALL_LLMS:
                if held not in seen:
                    cells.append((sweep, base, l1, l2, held, sorted(pairs_seeds[(l1, l2)])))
    return cells


def _metrics(preds_p, preds_u, u_targets):
    up, un = [], []
    for i in range(len(preds_u)):
        up.append(preds_u[i][u_targets[i] == 0][:, 0])
        un.append(preds_u[i][u_targets[i] == 1][:, 0])
    d = {}
    def add(name, fn):
        pt, lo, hi = bootstrap_metric(fn, up, un, n_bootstrap=N_BOOTSTRAP, cis=TEST_CIS)
        d[name] = pt
        for ci in hi:
            d[f'{name}_l_{ci}'] = lo[ci]; d[f'{name}_u_{ci}'] = hi[ci]
    add("auc", auc_fn); add("pos_prob", pos_prob_fn); add("neg_prob", neg_prob_fn)
    add("avg_pos_neg_prob", avg_prob_fn); add("tpr", tpr_fn); add("fnr", fnr_fn)
    add("tnr", tnr_fn); add("fpr", fpr_fn); add("plugin", plugin_fn)
    add("plugin-int", plugin_int_fn); add("entropy", binary_entropy_fn)
    add("entropy_pos", binary_entropy_pos_fn); add("entropy_neg", binary_entropy_neg_fn)
    add("bce", balanced_cross_entropy_fn)
    pt, lo, hi = bootstrap_metric_bbe(BBE_estimator, preds_p, preds_u, u_targets, n_bootstrap=N_BOOTSTRAP, cis=TEST_CIS)
    d["bbe"] = pt
    for ci in hi:
        d[f'bbe_l_{ci}'] = lo[ci]; d[f'bbe_u_{ci}'] = hi[ci]
    return d


def run_cell(idx):
    import torch
    from model_inference import get_preds_llm
    from models.conda import ConDADistilBert

    cells = enumerate_cells()
    assert 0 <= idx < len(cells), f"idx {idx} out of range 0..{len(cells)-1}"
    sweep, base, l1, l2, held, seeds = cells[idx]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_llm = f"{l1}|{l2}"
    test_llm = held.replace("_", " ")   # unseen LLM (spaced column name)
    print(f"[cell {idx}] sweep={sweep} train={train_llm} -> UNSEEN test={test_llm} seeds={seeds}")

    preds_p, preds_u, u_tgts = [], [], []
    for n in seeds:
        inner = Path(f"{base}/llm_type_{l1}|{l2}_{n}") / f"llm_type_{l1}|{l2}_{EPOCHS}"
        pts = [p for p in inner.iterdir() if p.is_file() and p.name.lower().endswith(".pt")]
        assert len(pts) == 1, f"expected 1 .pt in {inner}"
        net = ConDADistilBert(num_classes=2)
        sd = torch.load(pts[0], map_location=device)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        net.load_state_dict(sd); net.eval().to(device)
        # codex=True -> superset parquet serves every LLM's test set uniformly
        pp, upb, ut = get_preds_llm(DATA_TYPE, net, device, TEST_ALPHA, TEST_YEAR,
                                    test_llm, True, True, False, True, n, codex=True)
        preds_p.append(pp); preds_u.append(upb); u_tgts.append(ut)

    row = {
        "learning_method": "ConDA", "sweep": sweep, "data_type": DATA_TYPE,
        "train_alpha": TRAIN_ALPHA, "train_year": TRAIN_YEAR, "train_llm": train_llm,
        "seen_llm1": l1, "seen_llm2": l2, "unseen": True,
        "test_alpha": TEST_ALPHA, "test_year": TEST_YEAR, "test_llm": test_llm,
        "epochs": EPOCHS, "clean": True, "sentence": True, "gemini": False, "eval_flip": True,
        "cell_idx": idx,
    }
    row.update(_metrics(preds_p, preds_u, u_tgts))

    os.makedirs(CELL_DIR, exist_ok=True)
    out = f"{CELL_DIR}/cell_{idx:03d}.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"[cell {idx}] wrote {out}")


def merge():
    cells = enumerate_cells()
    frames = []
    missing = []
    for idx in range(len(cells)):
        f = f"{CELL_DIR}/cell_{idx:03d}.csv"
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
        else:
            missing.append(idx)
    if missing:
        print(f"WARNING: {len(missing)} missing cells: {missing}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"merged {len(df)} / {len(cells)} cells -> {OUTPUT_CSV}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "count"
    if mode == "count":
        print(len(enumerate_cells()))
    elif mode == "cell":
        run_cell(int(sys.argv[2]))
    elif mode == "merge":
        merge()
    else:
        raise SystemExit(f"unknown mode {mode}")
