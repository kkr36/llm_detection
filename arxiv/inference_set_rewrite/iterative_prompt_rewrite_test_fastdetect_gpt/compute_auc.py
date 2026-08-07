# conda env: any with pandas + scikit-learn (CPU only). No GPU needed.
#
# Headline attack metric. Fast-DetectGPT gives a scalar d(x) (HIGH=machine, LOW=human),
# so attack success = how SEPARABLE the rewritten-AI d's are from a human reference pool.
# AUC (AI labelled 1, human 0, score = d): ~1.0 = fully detectable, 0.5 = indistinguishable.
# The attacker drives AUC toward 0.5. We report the pre-attack baseline (original / mirror_0)
# for reference alongside the current timestep.
#
# Failed/refused rewrites (NaN d) are imputed to a large d so they count as DETECTED --
# a refusal is a failed attack and must not flatter the AUC.
#
# Usage:  python compute_auc.py --timestep 7 --split test --to_rewrite 50

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_ai_vs_human(ai_d, human_d):
    """AUC with AI=1 (score=d). Impute NaN AI scores to 'detected'; drop NaN humans."""
    human_d = np.asarray(human_d, dtype=float)
    human_d = human_d[np.isfinite(human_d)]
    ai_d = np.asarray(ai_d, dtype=float)
    if len(ai_d) == 0 or len(human_d) == 0:
        return float("nan"), 0, len(human_d)
    hi = np.nanmax(np.concatenate([ai_d, human_d])) + 10.0  # "most machine-like" sentinel
    ai_filled = np.where(np.isfinite(ai_d), ai_d, hi)
    scores = np.concatenate([ai_filled, human_d])
    labels = np.concatenate([np.ones(len(ai_filled)), np.zeros(len(human_d))])
    return roc_auc_score(labels, scores), len(ai_filled), len(human_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestep", type=int, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--to_rewrite", type=int, default=50)
    ap.add_argument("--human_ref", default="human_ref.csv")
    args = ap.parse_args()

    human = pd.read_csv(args.human_ref)["d"].to_numpy(dtype=float)
    stem = f"results_{args.timestep}_oss_{args.split}_{args.to_rewrite}_fastdetect.csv"
    df = pd.read_csv(stem)

    orig_d = df["original_d"].to_numpy(dtype=float)      # un-attacked AI baseline
    mirror_d = df[f"mirror_{args.timestep}_d"].to_numpy(dtype=float)

    base_auc, n_ai, n_h = auc_ai_vs_human(orig_d, human)
    cur_auc, _, _ = auc_ai_vs_human(mirror_d, human)

    print(f"== {stem}  (AI n={n_ai}, human n={n_h}) ==")
    print(f"baseline (original/mirror_0) AUC : {base_auc:.4f}   mean d={np.nanmean(orig_d):.3f}")
    print(f"attacked (mirror_{args.timestep}) AUC       : {cur_auc:.4f}   mean d={np.nanmean(mirror_d):.3f}")
    print(f"  human ref: mean d={np.nanmean(human):.3f}")
    print(f"  -> attack goal is AUC toward 0.5 (delta {cur_auc - base_auc:+.4f} vs baseline)")


if __name__ == "__main__":
    main()
