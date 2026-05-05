# must be run using conda env *llm_master*!

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/inference_set_rewrite/iterative_prompt_rewrite_test_pangram_y")
from pangram import Pangram

from prepare_metrics import (
    bootstrap_metric,
    auc_fn, pos_prob_fn, neg_prob_fn, avg_prob_fn,
    tpr_fn, fnr_fn, tnr_fn, fpr_fn,
    plugin_fn, plugin_int_fn,
    binary_entropy_fn, binary_entropy_pos_fn, binary_entropy_neg_fn,
)

# ── credentials & client ──────────────────────────────────────────────────────
with open("/home/kkr36/creds.json", "r") as fh:
    pangram_api_key = json.load(fh)["pangram_api_key"]
pangram_client = Pangram(api_key=pangram_api_key)

# ── paths & constants ─────────────────────────────────────────────────────────
DATA_PATH   = "/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_xyz_cs._10000_fronthalf.parquet"
RAW_OUT_DIR = "/share/garg/arxiv_kaggle/pangram_adversarial"
LOGGING_CSV = "logging_accuracy_xz.csv"
SAMPLE_SEED = 42
SAMPLE_N    = 2000
N_BOOTSTRAP = 2500
TEST_CIS    = [0.9, 0.95, 0.99]

os.makedirs(RAW_OUT_DIR, exist_ok=True)


# ── Pangram helpers ───────────────────────────────────────────────────────────
def predict_with_backoff(client, text, max_retries=5, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return client.predict(text)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    return {}


def parse_pangram_result(result):
    if result is None or len(result) == 0:
        return {k: np.nan for k in [
            "fraction_ai", "fraction_ai_assisted", "fraction_human",
            "num_ai_segments", "window_labels",
            "window_ai_assistance_scores", "window_confidences",
        ]}
    window_labels, window_ai_scores, window_confs = [], [], []
    for w in result.get("windows", []):
        window_labels.append(w.get("label", None))
        window_ai_scores.append(w.get("ai_assistance_score", np.nan))
        window_confs.append(w.get("confidence", np.nan))
    return {
        "fraction_ai":           result.get("fraction_ai", np.nan),
        "fraction_ai_assisted":  result.get("fraction_ai_assisted", np.nan),
        "fraction_human":        result.get("fraction_human", np.nan),
        "num_ai_segments":       result.get("num_ai_segments", np.nan),
        "window_labels":         window_labels,
        "window_ai_assistance_scores": window_ai_scores,
        "window_confidences":    window_confs,
    }


MAX_WORKERS = 14


def run_pangram_on_texts(texts, label):
    rows = [None] * len(texts)
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text) < 5 or "sorry" in text.lower():
                rows[i] = parse_pangram_result(None)
                rows[i]["text"] = text
                rows[i]["source"] = label
            else:
                futures[pool.submit(predict_with_backoff, pangram_client, text)] = i

        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Pangram on {label}")
        for future in pbar:
            i = futures[future]
            text = texts[i]
            parsed = parse_pangram_result(future.result())
            parsed["text"] = text
            parsed["source"] = label
            rows[i] = parsed

    return rows


# ── pangram score definitions ─────────────────────────────────────────────────
def score_avg_window(row):
    """Mean of window-level AI assistance scores."""
    scores = row["window_ai_assistance_scores"]
    if isinstance(scores, list) and len(scores) > 0:
        valid = [s for s in scores if isinstance(s, (int, float)) and not np.isnan(s)]
        return np.mean(valid) if valid else np.nan
    return np.nan


def score_fraction_ai(row):
    """Raw fraction_ai value from API."""
    return row["fraction_ai"]


def score_dominant_category(row):
    """1.0 if fraction_ai dominates, 0.5 if fraction_ai_assisted, 0.0 if fraction_human."""
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


# ── metrics logic ─────────────────────────────────────────────────────────────
def extract_scores(rows, score_fn):
    return [score_fn(r) for r in rows]


def clean_arr(scores):
    """Drop NaNs and return a float64 numpy array."""
    arr = np.array(scores, dtype=float)
    return arr[~np.isnan(arr)]


def update_dict(d, metric, point, lowers, uppers):
    d[metric] = point
    for ci in uppers:
        d[f"{metric}_l_{ci}"] = lowers[ci]
        d[f"{metric}_u_{ci}"] = uppers[ci]


def compute_metrics(ai_arr, human_arr, test_cis, n_bootstrap):
    """
    Run bootstrap_metric for all metrics that require two score distributions.
    ai_arr    : 1-D float64 numpy array of Pangram scores on AI/eval texts (preds_p)
    human_arr : 1-D float64 numpy array of Pangram scores on human texts  (preds_u)

    Mirrors the prepare_metrics.bootstrap_metric contract:
      preds_p_list = [ai_arr]     (one "seed")
      preds_u_list = [human_arr]

    Note: bbe is intentionally omitted — it requires a separately labeled
    positive set (known-AI training data) that Pangram evaluation does not produce.
    """
    preds_p_list = [ai_arr]
    preds_u_list = [human_arr]

    d = {}
    bm = lambda fn: bootstrap_metric(fn, preds_p_list, preds_u_list,
                                     n_bootstrap=n_bootstrap, cis=test_cis)

    update_dict(d, "auc",            *bm(auc_fn))
    update_dict(d, "pos_prob",        *bm(pos_prob_fn))
    update_dict(d, "neg_prob",        *bm(neg_prob_fn))
    update_dict(d, "avg_pos_neg_prob",*bm(avg_prob_fn))
    update_dict(d, "tpr",             *bm(tpr_fn))
    update_dict(d, "fnr",             *bm(fnr_fn))
    update_dict(d, "tnr",             *bm(tnr_fn))
    update_dict(d, "fpr",             *bm(fpr_fn))
    update_dict(d, "plugin",          *bm(plugin_fn))
    update_dict(d, "plugin-int",      *bm(plugin_int_fn))
    update_dict(d, "entropy",         *bm(binary_entropy_fn))
    update_dict(d, "entropy_pos",     *bm(binary_entropy_pos_fn))
    update_dict(d, "entropy_neg",     *bm(binary_entropy_neg_fn))
    return d


def compute_human_only_metrics(human_arr, test_cis, n_bootstrap):
    """
    For the human_abstract row we have no paired AI set, so only metrics that
    depend solely on preds_u are meaningful (neg_prob, entropy_neg).
    """
    preds_p_list = [human_arr]  # dummy; neg_prob_fn / entropy_neg_fn ignore preds_p
    preds_u_list = [human_arr]

    d = {}
    bm = lambda fn: bootstrap_metric(fn, preds_p_list, preds_u_list,
                                     n_bootstrap=n_bootstrap, cis=test_cis)
    update_dict(d, "neg_prob",    *bm(neg_prob_fn))
    update_dict(d, "entropy_neg", *bm(binary_entropy_neg_fn))
    return d


if __name__ == "__main__":
    # ── 1. load & sample data ─────────────────────────────────────────────────
    arxiv_data = pd.read_parquet(DATA_PATH)
    sample = arxiv_data.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).reset_index(drop=True)
    print(f"Sampled {len(sample)} rows")

    # ── 2. run Pangram on each eval set ───────────────────────────────────────
    human_rows     = run_pangram_on_texts(sample["human_abstract"].tolist(), "human")
    rewrite_x_rows = run_pangram_on_texts(sample["rewrite_X"].tolist(),      "rewrite_X")
    rewrite_z_rows = run_pangram_on_texts(sample["rewrite_Z"].tolist(),      "rewrite_Z")

    # ── 3. save raw predictions ───────────────────────────────────────────────
    raw_df = pd.DataFrame([
        {k: (str(v) if isinstance(v, list) else v) for k, v in r.items()}
        for r in human_rows + rewrite_x_rows + rewrite_z_rows
    ])
    raw_out_path = os.path.join(RAW_OUT_DIR, "pangram_adversarial_raw.csv")
    raw_df.to_csv(raw_out_path, index=False)
    print(f"Saved raw predictions → {raw_out_path}")

    # ── 4. build logging rows ─────────────────────────────────────────────────
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
                # Only metrics that need a single distribution are meaningful here.
                metrics = compute_human_only_metrics(human_arr, TEST_CIS, N_BOOTSTRAP)
            else:
                ai_arr = clean_arr(extract_scores(eval_rows, score_fn))
                metrics = compute_metrics(ai_arr, human_arr, TEST_CIS, N_BOOTSTRAP)

            base.update(metrics)
            new_rows.append(base)
            run_id += 1

    metrics_df = pd.concat([metrics_df, pd.DataFrame(new_rows)], ignore_index=True)
    metrics_df.to_csv(LOGGING_CSV, index=False)
    print(f"Appended {len(new_rows)} Pangram rows to {LOGGING_CSV}")
