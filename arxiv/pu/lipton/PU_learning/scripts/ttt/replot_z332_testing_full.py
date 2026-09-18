"""
Regenerate the SET2 (entire-batch) figs for every combo in the z332_testing grid, from the
already-merged per-combo samples CSVs on disk -- no GPU/model rerun needed, just replotting
with the current plot_ttt_curves.py (balanced accuracy + PNG+PDF). This mirrors exactly what
merge_ttt_curves.py writes at the end of each eval_ttt.sbatch job, so re-running it here is a
faithful "re-plot in place" for figs whose underlying data hasn't changed.

  python scripts/ttt/replot_z332_testing_full.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from plot_ttt_curves import make_curves, make_eda_pred_hist, make_ramp_plots
from scripts.ttt.plot_z332_z_only import Z332_TESTING_GRID

OUT_DIR = "ttt_logging/csv"
WINDOW = int(os.environ.get("TTT_WINDOW", "51"))


def main():
    n = 0
    for gran, cfg in Z332_TESTING_GRID.items():
        for bs in cfg["batches"]:
            for lr in cfg["lrs"]:
                figbase = f"figs/ttt/z332_testing/{gran}/batch_{bs}_lr_{lr}"
                for gtag in ("g00", "g50", "g100"):
                    tag = f"z332test_{gran}_b{bs}_lr{lr}_{gtag}"
                    samples_csv = f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_samples_{tag}.csv"
                    figdir = f"{figbase}/{gtag}"
                    if not os.path.exists(samples_csv):
                        print(f"  [missing] {samples_csv}")
                        continue
                    make_curves(samples_csv, figdir, window=WINDOW,
                                title_suffix=f"  ({gran}, rewrite_Z_332)")
                    make_eda_pred_hist(samples_csv, figdir)
                    make_ramp_plots(samples_csv, figdir)
                    n += 1
    print(f"done. re-plotted SET2 (entire batch) for {n} combos.")


if __name__ == "__main__":
    main()
