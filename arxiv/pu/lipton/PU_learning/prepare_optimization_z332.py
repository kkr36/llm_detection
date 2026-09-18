"""Evaluate Z332-trained TEDn/PNU checkpoints on rewrite_Z_332.

Writes separate CSVs so existing optimization logs and figures remain untouched:
  - logging_accuracy_xz_z332.csv       (TEDn xz332 -> rewrite_Z_332)
  - logging_accuracy_xz_PNU_z332.csv   (PNU Z_332 -> rewrite_Z_332)
  - logging_accuracy_xz_PN_z332.csv    (PN X -> rewrite_Z_332)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from estimator import BBE_estimator
from model_helper import get_model
from model_inference import get_preds_xy
from prepare_metrics import (
    auc_fn,
    avg_prob_fn,
    balanced_cross_entropy_fn,
    binary_entropy_fn,
    binary_entropy_neg_fn,
    binary_entropy_pos_fn,
    bootstrap_metric,
    bootstrap_metric_bbe,
    fnr_fn,
    fpr_fn,
    neg_prob_fn,
    plugin_fn,
    plugin_int_fn,
    pos_prob_fn,
    tnr_fn,
    tpr_fn,
)


PREDS_BASE = "/share/garg/arxiv_kaggle/predictions"
V2_PARQUET = "/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_xyz_v2_cs._10000_fronthalf.parquet"
os.environ["XY_EVAL_PARQUET"] = V2_PARQUET


def save_preds(path, pos_probs, unlabeled_probs, unlabeled_targets):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        pos_probs=pos_probs,
        unlabeled_probs=unlabeled_probs,
        unlabeled_targets=unlabeled_targets,
    )


def update_dict(metrics_dict, metric, point, lowers, uppers):
    metrics_dict[metric] = point
    for ci in uppers:
        assert ci in lowers
        metrics_dict[f"{metric}_l_{ci}"] = lowers[ci]
        metrics_dict[f"{metric}_u_{ci}"] = uppers[ci]


def get_metrics(preds_p, preds_u, u_targets, test_cis, n_bootstrap):
    preds_up_list, preds_un_list = [], []
    for i in range(len(preds_u)):
        preds_up = preds_u[i][u_targets[i] == 0][:, 0]
        preds_un = preds_u[i][u_targets[i] == 1][:, 0]
        preds_up_list.append(preds_up)
        preds_un_list.append(preds_un)

    metrics_dict = {}
    print("calculating metrics")
    update_dict(metrics_dict, "auc", *bootstrap_metric(auc_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "pos_prob", *bootstrap_metric(pos_prob_fn, preds_un_list, preds_up_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "neg_prob", *bootstrap_metric(neg_prob_fn, preds_un_list, preds_up_list, n_bootstrap=n_bootstrap, cis=test_cis))
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


def evaluate_config(cfg, output_csv):
    entrance_path = "logging_accuracy_xy"
    sentence = True
    clean = True
    epochs = 3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seeds = [0, 1, 2, 3, 4]
    test_alpha = 0.5
    test_cis = [0.9, 0.95, 0.99]

    if os.path.exists(output_csv):
        metrics_df = pd.read_csv(output_csv)
    else:
        metrics_df = pd.DataFrame()

    train_method = cfg["train_method"]
    train_alpha = cfg["train_alpha"]
    flip = cfg["flip"]
    llm = cfg["llm"]
    eval_llm_col = cfg["eval_llm"]

    duplicate_mask = (
        (not metrics_df.empty)
        and metrics_df["learning_method"].eq(train_method)
        & metrics_df["train_llm"].eq(llm)
        & metrics_df["eval_llm"].eq(eval_llm_col)
        & metrics_df["train_alpha"].eq(train_alpha)
    )
    if hasattr(duplicate_mask, "any") and duplicate_mask.any():
        print(f"Skipping existing row in {output_csv}: {train_method}, {llm}, {eval_llm_col}")
        return

    nets = {}
    alpha_dir = None
    for seed in seeds:
        if train_method == "PNU":
            model_dir = Path(f"{entrance_path}/normal_sentence/PNU/{seed}/{llm}/xy_{epochs}")
        else:
            model_dir = Path(f"{entrance_path}/normal_sentence/alpha_{train_alpha}/{seed}/{llm}/xy_{epochs}")
        pts = [
            p for p in model_dir.iterdir()
            if p.is_file() and p.name.lower().endswith(".pt") and train_method in p.name
        ]
        assert len(pts) == 1, f"Expected 1 {train_method} .pt for seed {seed}, found {len(pts)} in {model_dir}"
        alpha_dir = model_dir

        net = get_model("DistilBert")
        state_dict = torch.load(pts[0], map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)
        nets[seed] = net

    preds_p_list, preds_u_list, u_targets_list = [], [], []
    for seed, net in nets.items():
        pos_probs, unlabeled_probs, unlabeled_targets = get_preds_xy(
            net, device, test_alpha, flip, seed, sentence, clean, eval_llm_col
        )
        save_preds(
            f"{PREDS_BASE}/optimization_z332/{train_method}/{llm}/{eval_llm_col}/seed_{seed}.npz",
            pos_probs, unlabeled_probs, unlabeled_targets,
        )
        preds_p_list.append(pos_probs)
        preds_u_list.append(unlabeled_probs)
        u_targets_list.append(unlabeled_targets)

    run_id = 0 if metrics_df.empty or "run_id" not in metrics_df else int(pd.to_numeric(metrics_df["run_id"]).max()) + 1
    row = {
        "learning_method": train_method,
        "data_type": "xy",
        "train_llm": llm,
        "eval_llm": eval_llm_col,
        "train_alpha": train_alpha,
        "test_alpha": test_alpha,
        "flip": flip,
        "clean": clean,
        "sentence": sentence,
        "epochs": epochs,
        "model_dir": str(alpha_dir),
        "eval_parquet": V2_PARQUET,
        "run_id": run_id,
    }
    row.update(get_metrics(preds_p_list, preds_u_list, u_targets_list, test_cis, n_bootstrap=2500))
    metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
    metrics_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    evaluate_config(
        {"train_method": "TEDn", "train_alpha": 0.25, "flip": True, "llm": "xz332", "eval_llm": "rewrite_Z_332"},
        "logging_accuracy_xz_z332.csv",
    )
    evaluate_config(
        {"train_method": "PNU", "train_alpha": 0.25, "flip": True, "llm": "Z_332", "eval_llm": "rewrite_Z_332"},
        "logging_accuracy_xz_PNU_z332.csv",
    )
    evaluate_config(
        {"train_method": "PN", "train_alpha": 0, "flip": False, "llm": "X", "eval_llm": "rewrite_Z_332"},
        "logging_accuracy_xz_PN_z332.csv",
    )
