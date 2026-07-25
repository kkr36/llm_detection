"""Codex heatmap prep: extend the existing LLM detection matrix with Codex.

Same logic/metrics as prepare_heatmap.py, but instead of recomputing the whole
matrix it (1) carries over the already-computed 4x4 cells for the original LLMs
(dropping the out-of-date "all" rows/cols) and (2) computes only the *new* cells
that involve Codex:

    PN  block, Codex row   : Codex-trained PN detector   -> {orig 4 LLMs, Codex}  (5)
    PN  block, Codex column: each orig-LLM PN detector   -> Codex                 (4)
    PU  diagonal, Codex    : Codex-trained TEDn detector -> Codex                 (1)

Models are read from the canonical tree the codex trainer writes to:
    logging_accuracy_llm/normal_sentence/alpha_0/<LLM>_<seed>    (PN)
    logging_accuracy_llm/normal_sentence/alpha_0.5/<LLM>_<seed>  (TEDn/PU)

The test set for a given column is defined by the *test* LLM: the Codex column
reads the codex parquet (codex=True); every other column reads the standard
parquet (codex=False), so the original columns use the exact same test sets the
existing CSV used.

Output: logging_accuracy_llm_codex_remade.csv  (feeds plot_helper/plot_heatmaps_codex.py)
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np
from model_inference import get_preds_llm
from model_helper import *
from prepare_metrics import *
from estimator import BBE_estimator
import torch

# NOTE: prepare_heatmap.py runs its whole evaluation at module import (no
# __main__ guard), so we must NOT import from it. The three helpers below are
# copied verbatim from prepare_heatmap.py to keep identical metric behavior.
PREDS_BASE = "/share/garg/arxiv_kaggle/predictions"


def save_preds(path, pos_probs, unlabeled_probs, unlabeled_targets):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, pos_probs=pos_probs, unlabeled_probs=unlabeled_probs,
                        unlabeled_targets=unlabeled_targets)


def update_dict(metrics_dict, metric, point, lowers, uppers):
    metrics_dict[metric] = point
    for ci in uppers:
        assert (ci in lowers)
        metrics_dict[f'{metric}_l_{ci}'] = lowers[ci]
        metrics_dict[f'{metric}_u_{ci}'] = uppers[ci]


def get_metrics(preds_p, preds_u, u_targets, test_cis, n_bootstrap):
    preds_up_list, preds_un_list = [], []
    for i in range(len(preds_u)):
        preds_up = preds_u[i][u_targets[i] == 0][:, 0]
        preds_un = preds_u[i][u_targets[i] == 1][:, 0]
        preds_up_list.append(preds_up)
        preds_un_list.append(preds_un)

    metrics_dict = {}
    print('calculating metrics')
    update_dict(metrics_dict, "auc", *bootstrap_metric(auc_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "pos_prob", *bootstrap_metric(pos_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "neg_prob", *bootstrap_metric(neg_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "avg_pos_neg_prob", *bootstrap_metric(avg_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "tpr", *bootstrap_metric(tpr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "fnr", *bootstrap_metric(fnr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "tnr", *bootstrap_metric(tnr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "fpr", *bootstrap_metric(fpr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "plugin", *bootstrap_metric(plugin_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "plugin-int", *bootstrap_metric(plugin_int_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "entropy", *bootstrap_metric(binary_entropy_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "entropy_pos", *bootstrap_metric(binary_entropy_pos_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "entropy_neg", *bootstrap_metric(binary_entropy_neg_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "bce", *bootstrap_metric(balanced_cross_entropy_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "bbe", *bootstrap_metric_bbe(BBE_estimator, preds_p, preds_u, u_targets, n_bootstrap=n_bootstrap, cis=test_cis))
    return metrics_dict

# ---------------- switches (mirror prepare_heatmap.py) ----------------
entrance_path = "logging_accuracy_llm"
data_type = "ArXiv_BERT"
sentence = True
clean = True
gemini = False
epochs = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_year = 2020
test_year = 2020
seeds = 5
test_alpha = 0.5
test_cis = [.9, .95, .99]
n_bootstrap = 2500

ORIG_LLMS = ["Qwen", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct"]
CODEX = "Codex"
ALL_LLMS = ORIG_LLMS + [CODEX]

EXISTING_CSV = "logging_accuracy_llm_remade.csv"
output_csv = "logging_accuracy_llm_codex_remade.csv"

# (learning_method, train_alpha) for each method name
METHOD_ALPHA = {"PN": 0, "PU": 0.5}


def load_nets(train_llm_name, train_alpha):
    """Load the 5 seed detectors for (train_llm, train_alpha) from the canonical tree."""
    nets, model_path = [], None
    for n in range(seeds):
        alpha_dir = Path(
            f"{entrance_path}/normal_sentence/alpha_{train_alpha}/{train_llm_name}_{n}/"
            f"llm_type_{train_llm_name}_{epochs}"
        )
        pts = [p for p in alpha_dir.iterdir()
               if p.is_file() and p.name.lower().endswith(".pt")]
        assert len(pts) == 1, f"expected 1 .pt in {alpha_dir}, found {len(pts)}"
        model_path = str(pts[0])
        net = get_model("DistilBert")
        state_dict = torch.load(pts[0], map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)
        nets.append(net)
    return nets, model_path


def eval_cell(model_name, train_llm, test_llm):
    """Compute one (train_llm, test_llm) cell, averaging over 5 seeds; returns a row dict."""
    train_alpha = METHOD_ALPHA[model_name]
    train_llm_name = train_llm.replace(" ", "_")
    test_llm_name = test_llm.replace(" ", "_")
    eval_flip = True  # human=positive for eval (matches prepare_heatmap.py)
    use_codex = (test_llm == CODEX)  # the codex parquet is only needed for the Codex column

    print(f"train: {model_name} {train_llm} (alpha={train_alpha}) | test: {test_llm} "
          f"(codex_parquet={use_codex})")

    nets, model_path = load_nets(train_llm_name, train_alpha)

    preds_p_list, preds_u_list, u_targets_list = [], [], []
    for n in range(seeds):
        pos_probs, unlabeled_probs, unlabeled_targets = get_preds_llm(
            data_type, nets[n], device, test_alpha, test_year, test_llm,
            sentence, clean, gemini, eval_flip, n, codex=use_codex,
        )
        # PN model outputs P(LLM); negate to get P(human), mirroring prepare_heatmap.py
        if model_name == "PN":
            pos_probs = 1 - pos_probs
            unlabeled_probs = 1 - unlabeled_probs
        save_preds(
            f"{PREDS_BASE}/heatmap_codex/{model_name}/train_{train_llm_name}/"
            f"alpha_{train_alpha}/test_{test_llm_name}/seed_{n}.npz",
            pos_probs, unlabeled_probs, unlabeled_targets,
        )
        preds_p_list.append(pos_probs)
        preds_u_list.append(unlabeled_probs)
        u_targets_list.append(unlabeled_targets)

    info = {
        "learning_method": model_name,
        "data_type": data_type,
        "train_alpha": train_alpha,
        "train_year": train_year,
        "train_llm": train_llm,
        "test_alpha": test_alpha,
        "test_year": test_year,
        "test_llm": test_llm,
        "epochs": epochs,
        "clean": clean,
        "sentence": sentence,
        "gemini": gemini,
        "eval_flip": eval_flip,
        "model_path": model_path,
    }
    metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list,
                          test_cis=test_cis, n_bootstrap=n_bootstrap)
    row = {}
    row.update(info)
    row.update(metrics)
    return row


def build_cell_list():
    """The set of cells needed for the Codex heatmap (no 'all')."""
    cells = []  # (model_name, train_llm, test_llm)
    # PN block: full 5x5 over {orig 4, Codex}
    for train_llm in ALL_LLMS:
        for test_llm in ALL_LLMS:
            cells.append(("PN", train_llm, test_llm))
    # PU (TEDn) diagonal for each LLM (only the diagonal is used by the plot)
    for llm in ALL_LLMS:
        cells.append(("PU", llm, llm))
    return cells


if __name__ == "__main__":
    # ---- seed the output with the already-computed original cells (drop 'all') ----
    if os.path.exists(output_csv):
        metrics_df = pd.read_csv(output_csv)
    elif os.path.exists(EXISTING_CSV):
        prev = pd.read_csv(EXISTING_CSV)
        metrics_df = prev[(prev["train_llm"] != "all") & (prev["test_llm"] != "all")].copy()
        metrics_df.to_csv(output_csv, index=False)
        print(f"seeded {output_csv} with {len(metrics_df)} carried-over (non-'all') cells")
    else:
        metrics_df = pd.DataFrame()

    def already_done(model_name, train_llm, test_llm):
        if len(metrics_df) == 0:
            return False
        m = (
            (metrics_df["learning_method"] == model_name)
            & (metrics_df["train_llm"] == train_llm)
            & (metrics_df["test_llm"] == test_llm)
        )
        return bool(m.any())

    for model_name, train_llm, test_llm in build_cell_list():
        if already_done(model_name, train_llm, test_llm):
            print(f"skip (already present): {model_name} {train_llm} -> {test_llm}")
            continue
        row = eval_cell(model_name, train_llm, test_llm)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        metrics_df.to_csv(output_csv, index=False)  # crash-safe: save after every cell

    print(f"done -> {output_csv} ({len(metrics_df)} rows)")
