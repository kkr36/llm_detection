### DO NOT OVERWRITE, EDIT, OR TOUCH ANYTHING IN THIS FILE ###
# must be run using conda env *llm_embeddings* on a GPU node!
#
# Phase 2 of the Fast-DetectGPT attack loop: SCORE the rewritten mirrors.
# Fast-DetectGPT emits a scalar conditional-probability curvature d(x) per text
# (HIGH = machine, LOW = human). We score each WHOLE abstract (granularity="abstract";
# sentence-level d is ~useless here, AUC ~0.5). Lower d / lower sigmoid(d) = more
# human-like = better attack. The headline attack metric is AUC vs. a human
# reference pool -- see compute_auc.py; this file just produces the per-abstract d.
#
# Reads   results_{timestep}_oss_{split}_{to_rewrite}.csv   (mirror_{timestep} column)
#         results_0_oss_{split}_{to_rewrite}_fastdetect.csv (baseline original_* columns)
# Writes  results_{timestep}_oss_{split}_{to_rewrite}_fastdetect.csv  adding
#         original_d, original_score_avg, mirror_{timestep}_d, mirror_{timestep}_score_avg

import argparse
import sys

import numpy as np
import pandas as pd

from util import clean_text

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from fastdetect import FastDetectScorer, ScoreCache, cache_slug

# ---- Fast-DetectGPT config (swap here to change the target detector) ----
SCORING_MODEL = "EleutherAI/gpt-neo-2.7B"   # log p (p_theta)
SAMPLING_MODEL = "EleutherAI/gpt-j-6B"      # sampling q (q_phi); two-model black-box pair
GRANULARITY = "abstract"                      # NOT sentence -- sentence-level AUC ~0.5
BATCH_SIZE = 8                                # gpt-j + neo fit on one A6000 at bs=8


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_abstracts(texts, scorer=None, cache=None):
    """Return d for each text (NaN for invalid/empty), using the shared sha1 cache.

    A single FastDetectScorer/ScoreCache can be threaded in to score several files
    in one process; otherwise they are created lazily here.
    """
    # clean_text takes/returns a list; match the preprocessing used for human_ref.
    valid = [isinstance(t, str) and len(t.strip()) >= 5 for t in texts]
    raw = [t if v else "" for t, v in zip(texts, valid)]
    cleaned = clean_text(raw)

    if cache is None:
        cache = ScoreCache(cache_slug(SCORING_MODEL, SAMPLING_MODEL), GRANULARITY)
    to_score = [c for c, v in zip(cleaned, valid) if v]
    todo = cache.missing(to_score)
    if todo:
        if scorer is None:
            scorer = FastDetectScorer(
                SCORING_MODEL, sampling_model_name=SAMPLING_MODEL,
                device="cuda", batch_size=BATCH_SIZE)
        scores = scorer.score_texts(todo)
        cache.update(todo, scores)
        cache.save()

    d = cache.lookup(cleaned)
    d = np.where(valid, d, np.nan)   # invalid/refused rewrites -> NaN (handled in compute_auc)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--to_rewrite", type=int, default=50)
    args = parser.parse_args()

    split, timestep, to_rewrite = args.split, args.timestep, args.to_rewrite
    input_stem = f"results_{timestep}_oss_{split}_{to_rewrite}"
    input_data = pd.read_csv(f"{input_stem}.csv")
    baseline = pd.read_csv(f"results_0_oss_{split}_{to_rewrite}_fastdetect.csv")

    mirrors = input_data[f"mirror_{timestep}"].tolist()
    d = score_abstracts(mirrors)

    input_data["original_d"] = baseline["original_d"].tolist()
    input_data["original_score_avg"] = baseline["original_score_avg"].tolist()
    input_data[f"mirror_{timestep}_d"] = d
    input_data[f"mirror_{timestep}_score_avg"] = sigmoid(d)   # P(AI); lower = more human

    out = f"{input_stem}_fastdetect.csv"
    input_data.to_csv(out, index=False)

    finite = d[np.isfinite(d)]
    print(f"wrote {out}")
    print(f"  mirror_{timestep}: mean d={np.nanmean(d):.3f}  "
          f"mean sigmoid(d)={np.nanmean(sigmoid(d)):.3f}  "
          f"({len(finite)}/{len(d)} finite)")
    print(f"  original (t0):     mean d={np.nanmean(input_data['original_d']):.3f}")
