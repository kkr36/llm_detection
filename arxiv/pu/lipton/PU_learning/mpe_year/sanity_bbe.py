"""
BBE sanity checks for the PU models (mpe_year).

For each year, using every trained model:
  (a) mean p_probs on the known-positive P-set (val-split AI mirror chunks) --
      should be ~1.0 (confirms class 0 = positive = AI, and P scores high).
  (b) build a TEST unlabeled pool that is exactly `alpha` full AI rewrites
      (eval-row mirror chunks, held out from training) + (1-alpha) human eval
      chunks, and check BBE recovers ~alpha. Default alpha=0.25.

If (a)~1 and (b)~0.25, BBE is working and the earlier low estimate on pure-human
test chunks is a genuine P/U distribution effect, not a labeling bug.

Needs a GPU. Run:
    sbatch mpe_year/run_sanity.sbatch      # or: python mpe_year/sanity_bbe.py
"""

import os
import sys
import glob
import argparse

PU_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PU_ROOT not in sys.path:
    sys.path.insert(0, PU_ROOT)

import numpy as np
import pandas as pd
import torch

from estimator import p_probs, u_probs, BBE_estimator
from mpe_year.mpe_data import (
    YEARS, read_fronthalf_pu, eval_human_chunks, eval_ai_mirror_chunks, EVAL_SEED,
)
from mpe_year.eval_pu import make_p_loader, make_u_loader, load_net

DEFAULT_LOG_DIR = "/share/garg/arxiv_kaggle/mpe_year_models"


def build_alpha_u(year, alpha, n_total=1000, seed=EVAL_SEED):
    """Test unlabeled pool: alpha full-AI-rewrite chunks + (1-alpha) human chunks."""
    human = eval_human_chunks(year)                 # shared sampled human chunks
    ai = eval_ai_mirror_chunks(year)                # held-out eval-row AI mirror chunks
    n_ai = int(round(alpha * n_total))
    n_hu = n_total - n_ai
    rng = np.random.default_rng(seed)
    u_ai = list(rng.choice(ai, size=min(n_ai, len(ai)), replace=False))
    u_hu = list(rng.choice(human, size=min(n_hu, len(human)), replace=False))
    return u_ai + u_hu, len(u_ai) / (len(u_ai) + len(u_hu))


def run_year(year, log_dir, net_type, alpha, seed_for_p, device):
    model_paths = sorted(glob.glob(os.path.join(log_dir, str(year), "*.pt")))
    if not model_paths:
        print(f"[{year}] no models; skip")
        return []

    # P-set: val-split AI mirror chunks (known positives)
    p_texts, p_labels = read_fronthalf_pu(year, alpha=0.0, split="val", seed=seed_for_p)
    p_texts = [t for t, l in zip(p_texts, p_labels) if l == 1]
    # test U with true alpha injected full AI rewrites
    u_texts, true_alpha = build_alpha_u(year, alpha)
    print(f"[{year}] |P|={len(p_texts)}  |U|={len(u_texts)}  true_alpha={true_alpha:.3f}")

    p_loader = make_p_loader(p_texts)
    u_loader = make_u_loader(u_texts)

    rows = []
    for mp in model_paths:
        net = load_net(mp, net_type, device)
        pos_probs = p_probs(net, device, p_loader)
        unlabeled_probs, unlabeled_targets = u_probs(net, device, u_loader)
        mean_p_on_P = float(np.mean(pos_probs))                 # (a) expect ~1.0
        bbe, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)  # (b) expect ~alpha
        mean_ai_on_U = float(np.mean(unlabeled_probs[:, 0]))
        rows.append({"year": year, "true_alpha": round(true_alpha, 3),
                     "mean_pprob_on_P": mean_p_on_P, "bbe_on_alpha_U": float(bbe),
                     "mean_ai_prob_on_U": mean_ai_on_U, "model": os.path.basename(mp)})
        print(f"  {os.path.basename(mp)}: mean P(AI) on P={mean_p_on_P:.3f}  "
              f"BBE={bbe:.3f}  meanP(AI) on U={mean_ai_on_U:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--net-type", default="DistilBert")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--seed-for-p", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_rows = []
    for year in YEARS:
        all_rows += run_year(year, args.log_dir, args.net_type, args.alpha, args.seed_for_p, device)

    df = pd.DataFrame(all_rows)
    out = os.path.join(PU_ROOT, "mpe_year", "results", f"sanity_bbe_alpha{args.alpha}.csv")
    df.to_csv(out, index=False)
    summary = (df.groupby("year")
               .agg(true_alpha=("true_alpha", "first"),
                    mean_pprob_on_P=("mean_pprob_on_P", "mean"),
                    bbe_on_alpha_U=("bbe_on_alpha_U", "mean"),
                    mean_ai_prob_on_U=("mean_ai_prob_on_U", "mean"))
               .reset_index())
    print("\n=== summary (averaged over seeds) ===")
    print(summary.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
