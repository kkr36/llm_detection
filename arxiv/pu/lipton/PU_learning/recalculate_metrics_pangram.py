# must be run using conda env *llm_master*!
#
# Reads an already-saved raw Pangram predictions CSV and recomputes all
# metrics (including the new BCE metric) without calling the Pangram API.
# Appends new rows to LOGGING_CSV.

import ast
import os
import sys
import numpy as np
import pandas as pd

from prepare_metrics import (
    bootstrap_metric,
    auc_fn, pos_prob_fn, neg_prob_fn, avg_prob_fn,
    tpr_fn, fnr_fn, tnr_fn, fpr_fn,
    plugin_fn, plugin_int_fn,
    binary_entropy_fn, binary_entropy_pos_fn, binary_entropy_neg_fn,
    balanced_cross_entropy_fn,
)

# ── paths & constants (must match the original run) ──────────────────────────
DATA_PATH   = "/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_xyz_cs._10000_fronthalf.parquet"
RAW_OUT_DIR = "/share/garg/arxiv_kaggle/pangram_adversarial"
RAW_CSV     = os.path.join(RAW_OUT_DIR, "pangram_adversarial_raw.csv")
LOGGING_CSV = "logging_accuracy_xz.csv"
SAMPLE_N    = 2000
SAMPLE_SEED = 42
N_BOOTSTRAP = 2500
TEST_CIS    = [0.9, 0.95, 0.99]


# ── score definitions (identical to original script) ─────────────────────────
def score_avg_window(row):
    scores = row["window_ai_assistance_scores"]
    if isinstance(scores, list) and len(scores) > 0:
        valid = [s for s in scores if isinstance(s, (int, float)) and not np.isnan(s)]
        return np.mean(valid) if valid else np.nan
    return np.nan

def score_fraction_ai(row):
    return row["fraction_ai"]

def score_dominant_category(row):
    fa, fasst, fh = row["fraction_ai"], row["fraction_ai_assisted"], row["fraction_human"]
    if any(isinstance(v, float) and np.isnan(v) for v in [fa, fasst, fh]):
        return np.nan
    if fa >= fasst and fa >= fh:
        return 1.0
    if fasst >= fa and fasst >= fh:
        return 0.5
    return 0.0

SCORE_DEFS = {
    "avg_window_ai_assistance": score_avg_window,
    "fraction_ai":              score_fraction_ai,
    "dominant_category":        score_dominant_category,
}


# ── helpers ───────────────────────────────────────────────────────────────────
def parse_list_col(val):
    """Restore a list column that was serialized with str()."""
    if isinstance(val, list):
        return val
    if isinstance(val, float) and np.isnan(val):
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        return []

LIST_COLS = ["window_labels", "window_ai_assistance_scores", "window_confidences"]

def df_to_rows(df):
    rows = []
    for _, r in df.iterrows():
        d = r.to_dict()
        for col in LIST_COLS:
            d[col] = parse_list_col(d.get(col, []))
        rows.append(d)
    return rows

def extract_scores(rows, score_fn):
    return [score_fn(r) for r in rows]

def clean_arr(scores):
    arr = np.array(scores, dtype=float)
    return arr[~np.isnan(arr)]

def update_dict(d, metric, point, lowers, uppers):
    d[metric] = point
    for ci in uppers:
        d[f"{metric}_l_{ci}"] = lowers[ci]
        d[f"{metric}_u_{ci}"] = uppers[ci]


# ── metric computation ────────────────────────────────────────────────────────
def compute_metrics(ai_arr, human_arr, test_cis, n_bootstrap):
    preds_p_list = [ai_arr]
    preds_u_list = [human_arr]
    d = {}
    bm = lambda fn: bootstrap_metric(fn, preds_p_list, preds_u_list,
                                     n_bootstrap=n_bootstrap, cis=test_cis)
    update_dict(d, "auc",             *bm(auc_fn))
    update_dict(d, "pos_prob",         *bm(pos_prob_fn))
    update_dict(d, "neg_prob",         *bm(neg_prob_fn))
    update_dict(d, "avg_pos_neg_prob", *bm(avg_prob_fn))
    update_dict(d, "tpr",              *bm(tpr_fn))
    update_dict(d, "fnr",              *bm(fnr_fn))
    update_dict(d, "tnr",              *bm(tnr_fn))
    update_dict(d, "fpr",              *bm(fpr_fn))
    update_dict(d, "plugin",           *bm(plugin_fn))
    update_dict(d, "plugin-int",       *bm(plugin_int_fn))
    update_dict(d, "entropy",          *bm(binary_entropy_fn))
    update_dict(d, "entropy_pos",      *bm(binary_entropy_pos_fn))
    update_dict(d, "entropy_neg",      *bm(binary_entropy_neg_fn))
    update_dict(d, "bce",              *bm(balanced_cross_entropy_fn))
    return d

def compute_human_only_metrics(human_arr, test_cis, n_bootstrap):
    preds_p_list = [human_arr]
    preds_u_list = [human_arr]
    d = {}
    bm = lambda fn: bootstrap_metric(fn, preds_p_list, preds_u_list,
                                     n_bootstrap=n_bootstrap, cis=test_cis)
    update_dict(d, "neg_prob",    *bm(neg_prob_fn))
    update_dict(d, "entropy_neg", *bm(binary_entropy_neg_fn))
    return d


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. load raw predictions
    raw_df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(raw_df)} raw rows from {RAW_CSV}")

    human_rows     = df_to_rows(raw_df[raw_df["source"] == "human"].reset_index(drop=True))
    rewrite_x_rows = df_to_rows(raw_df[raw_df["source"] == "rewrite_X"].reset_index(drop=True))
    rewrite_z_rows = df_to_rows(raw_df[raw_df["source"] == "rewrite_Z"].reset_index(drop=True))
    print(f"  human={len(human_rows)}, rewrite_X={len(rewrite_x_rows)}, rewrite_Z={len(rewrite_z_rows)}")

    # 2. load or create logging CSV
    if os.path.exists(LOGGING_CSV):
        metrics_df = pd.read_csv(LOGGING_CSV)
    else:
        metrics_df = pd.DataFrame()
    run_id = len(metrics_df)

    eval_sets = [
        ("human_abstract", human_rows),
        ("rewrite_X",      rewrite_x_rows),
        ("rewrite_Z",      rewrite_z_rows),
    ]

    new_rows = []
    for score_name, score_fn in SCORE_DEFS.items():
        human_arr = clean_arr(extract_scores(human_rows, score_fn))

        for eval_col, eval_rows in eval_sets:
            base = {
                "learning_method":     "pangram",
                "data_type":           "pangram",
                "train_llm":           np.nan,
                "eval_llm":            eval_col,
                "pangram_score_type":  score_name,
                "train_alpha":         np.nan,
                "test_alpha":          np.nan,
                "flip":                np.nan,
                "clean":               np.nan,
                "sentence":            np.nan,
                "epochs":              np.nan,
                "model_dir":           DATA_PATH,
                "run_id":              run_id,
                "pangram_sample_n":    SAMPLE_N,
                "pangram_sample_seed": SAMPLE_SEED,
            }

            if eval_col == "human_abstract":
                metrics = compute_human_only_metrics(human_arr, TEST_CIS, N_BOOTSTRAP)
            else:
                ai_arr = clean_arr(extract_scores(eval_rows, score_fn))
                metrics = compute_metrics(ai_arr, human_arr, TEST_CIS, N_BOOTSTRAP)

            base.update(metrics)
            new_rows.append(base)
            run_id += 1

    metrics_df = pd.concat([metrics_df, pd.DataFrame(new_rows)], ignore_index=True)
    metrics_df.to_csv(LOGGING_CSV, index=False)
    print(f"Appended {len(new_rows)} rows to {LOGGING_CSV}")
