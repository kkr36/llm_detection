"""
Merge the per-mode outputs written by eval_TEDn_X_rewrite_Z_ttt.py (one process per
TTT mode, run concurrently on separate GPUs) and draw the continual-learning curves.

Reads   logging_accuracy_xz_TTT_rewrite_Z_samples_<mode>.csv   (per-mode, per-sample)
        logging_accuracy_xz_TTT_rewrite_Z_metrics_<mode>.csv   (per-mode, aggregate)
Writes  logging_accuracy_xz_TTT_rewrite_Z_samples.csv          (combined per-sample)
        logging_accuracy_xz_TTT_rewrite_Z.csv                  (combined aggregate metrics)
        figs/ttt/<gran>/ttt_curve_*.png

Standalone: python merge_ttt_curves.py
"""
import os
import glob
import pandas as pd
from plot_ttt_curves import make_curves, make_eda_pred_hist, make_ramp_plots

GRAN = os.environ.get("TTT_GRAN", "abstract")
EVAL_COL = os.environ.get("TTT_EVAL_COL", "rewrite_Z")
WINDOW = int(os.environ.get("TTT_WINDOW", "51"))
OUT_DIR = "ttt_logging/csv"       # all TTT logging CSVs live here

sample_files = sorted(glob.glob(f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_samples_{GRAN}_*.csv"))
assert sample_files, f"no per-mode samples CSVs found for gran={GRAN} (run the eval first)"
samples = pd.concat([pd.read_csv(f) for f in sample_files], ignore_index=True)
samples_csv = f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_samples_{GRAN}.csv"
samples.to_csv(samples_csv, index=False)
print(f"[{GRAN}] merged {len(sample_files)} mode sample-files -> {samples_csv} "
      f"({len(samples)} rows, strategies={sorted(samples['strategy'].unique())})")

metric_files = sorted(glob.glob(f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_metrics_{GRAN}_*.csv"))
if metric_files:
    metrics = pd.concat([pd.read_csv(f) for f in metric_files], ignore_index=True)
    metrics_csv = f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_metrics_{GRAN}.csv"
    metrics.to_csv(metrics_csv, index=False)
    cols = [c for c in ["ttt_mode", "auc", "stream_accuracy"] if c in metrics.columns]
    print(f"merged metrics -> {metrics_csv}\n{metrics[cols].to_string(index=False)}")

figs_dir = os.environ.get("TTT_FIGDIR") or f"figs/ttt/{GRAN}"
make_curves(samples_csv, figs_dir, window=WINDOW, title_suffix=f"  ({GRAN}, {EVAL_COL})")
make_eda_pred_hist(samples_csv, figs_dir,
                   modes=tuple(sorted(samples["strategy"].unique())),
                   title_suffix=f"  ({GRAN}, {EVAL_COL})")
make_ramp_plots(samples_csv, figs_dir, title_suffix=f"  ({GRAN})")
print(f"wrote curves + eda (+ ramp if applicable) -> {figs_dir}/")
