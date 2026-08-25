"""
James's method (word-occurrence MLE mixture-proportion estimator) for mpe_year.

Train ONCE on 2020 (first 8,000 rows):
    human sentences  vs  AI-mirror sentences  -> estimate_text_distribution (logP/logQ)
Evaluate the single 2020-trained word distribution on EACH year's test set
(last 2,000 rows' human_abstract, years 2020/2023/2025):
    MLE.inference over the eval human *sentences* -> alpha_hat = estimated AI fraction.

Everything heavy is imported: sentence splitting (data_helper.IMDb.split_into_sentences,
via mpe_data), and estimate_text_distribution / MLE from james_methods. This file only
chains them and reproduces the point/CI aggregation used by model_inference.MLE_james.

Env: llm_embeddings (has spaCy en_core_web_lg + swifter). CPU-only.
Run:
    cd .../PU_learning
    /home/kkr36/.conda/envs/llm_embeddings/bin/python mpe_year/run_james.py
"""

import os
import sys

PU_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PU_ROOT not in sys.path:
    sys.path.insert(0, PU_ROOT)

import numpy as np
import pandas as pd

from james_methods import estimate_text_distribution, MLE
from mpe_year.mpe_data import (
    YEARS, load_split, train_human_ai_abstracts,
    chunk_tokens_of, eval_human_chunks, tokens_of,
)

OUT_DIR = os.path.join(PU_ROOT, "mpe_year", "results")
# James's trained artifact (the word log-prob distribution) is saved under /share,
# parallel to the PU .pt models in /share/garg/arxiv_kaggle/mpe_year_models/{year}/.
WORD_DIR = "/share/garg/arxiv_kaggle/mpe_year_models/james"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WORD_DIR, exist_ok=True)

N_BOOTSTRAP = 2500
TEST_CIS = [0.9, 0.95, 0.99]
TRAIN_YEAR = 2020  # James's word distribution is trained once, on 2020


def aggregate(estimates, test_cis):
    """Point estimate + CI half-widths, exactly as model_inference.MLE_james does."""
    cis = {}
    for ci in test_cis:
        diff = (1.0 - ci) / 2
        cis[ci] = np.percentile(estimates, [diff, 1.0 - diff])
    solution = None
    half_widths = {}
    for ci in test_cis:
        lo, hi = cis[ci]
        if ci == max(test_cis):
            solution = round(float(np.mean([lo, hi])), 3)
        half_widths[ci] = round(float((hi - lo) / 2), 3)
    return solution, half_widths


def train_word_dist():
    """Estimate the 2020 human-vs-AI word log-prob distribution (train once)."""
    train_df, _ = load_split(TRAIN_YEAR)
    human_abs, ai_abs = train_human_ai_abstracts(train_df)
    print(f"[train {TRAIN_YEAR}] abstracts: human={len(human_abs)} ai_mirror={len(ai_abs)}")
    # each data point = one 75-word chunk, tokenized to a word list
    human_df = pd.DataFrame({"human_sentence": chunk_tokens_of(human_abs)})
    ai_df = pd.DataFrame({"ai_sentence": chunk_tokens_of(ai_abs)})

    word_path = os.path.join(WORD_DIR, f"james_logprob_{TRAIN_YEAR}.parquet")
    estimate_text_distribution(human_df, ai_df, word_path)
    return word_path


def eval_year(mle, word_path, year):
    """Apply the (2020-trained) MLE to one year's held-out human abstracts."""
    eval_chunks = eval_human_chunks(year)          # the shared sampled 1,000 chunk strings
    eval_chunk_tokens = tokens_of(eval_chunks)     # tokenize the SAME chunks for James
    inf_df = pd.DataFrame({"inference_sentence": eval_chunk_tokens})
    print(f"[eval {year}] scored chunks: {len(eval_chunk_tokens)}")

    estimates = mle.inference(inf_df, exploded_data=True, n_bootstrap=N_BOOTSTRAP, test_cis=TEST_CIS)
    mpe, half_widths = aggregate(estimates, TEST_CIS)

    row = {
        "method": "james",
        "train_year": TRAIN_YEAR,
        "year": year,
        "mpe": mpe,
        "mpe_mean": round(float(np.mean(estimates)), 4),
        "n_eval_chunks": len(eval_chunk_tokens),
        "n_bootstrap": N_BOOTSTRAP,
        "word_dist_path": word_path,
    }
    for ci in TEST_CIS:
        row[f"mpe_l_{ci}"] = round(mpe - half_widths[ci], 3)
        row[f"mpe_u_{ci}"] = round(mpe + half_widths[ci], 3)
    print(f"[{year}] James MPE (AI fraction) = {mpe}  (mean={row['mpe_mean']})")
    return row


def main():
    word_path = train_word_dist()
    mle = MLE(word_path)  # load the 2020 word distribution once, reuse for every test year
    rows = [eval_year(mle, word_path, y) for y in YEARS]
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "james_mpe.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(df[["method", "train_year", "year", "mpe", "mpe_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
