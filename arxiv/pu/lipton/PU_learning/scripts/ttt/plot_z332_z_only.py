"""
Produce the "rewrite_Z_332 only" figure set (SET 1) for the z332 high-lr sentence ramp.

Two figure sets are wanted per g-value:
  SET 2  entire batch      -- the full make_curves suite on the whole mixed stream
                              (human + all AI, X and Z). This is produced UNCHANGED by
                              eval_ttt.sbatch's merge step into  <figdir>/  (exact same
                              output as the earlier ramp runs). This script does NOT touch it.
  SET 1  rewrite_Z_332 only -- the SAME curve suite, but restricted to {human + the
                              rewrite_Z_332 AI mirrors}, written to  <figdir>/z332_only/ .
                              g00 has no Z (growth=0), so its restricted set is on rewrite_X
                              instead -- the only AI data present -- per the agreed handling.

Restriction is a row filter on the merged samples CSV (label==human OR ai_source==SRC),
then the identical make_curves / make_eda_pred_hist / make_ramp_plots are re-run on it, so
SET 1 mirrors SET 2 curve-for-curve. Because Z_332 ramps in gradually, g50's early batches
have no Z rows and simply don't appear in the per-batch Z curves -- expected.

Caveat: the per-row `correct_cal` column was calibrated on the FULL stream's P(AI) median
(computed in eval_TEDn_X_rewrite_Z_ttt.py), so SET 1's *calibrated*-accuracy curve inherits
the full-stream threshold. AUC, fixed-0.5 accuracy, and recall are unaffected by this.

  python scripts/ttt/plot_z332_z_only.py      # run after the 3 eval jobs finish (default prefix/figbase)
  python scripts/ttt/plot_z332_z_only.py <tag_prefix> <figbase>   # another 3-job ramp experiment
  python scripts/ttt/plot_z332_z_only.py z332_testing              # entire z332_testing batch/lr/gran grid
"""
import os
import sys

# Make the PU_learning repo root importable regardless of cwd (this file lives in scripts/ttt).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from plot_ttt_curves import make_curves, make_eda_pred_hist, make_ramp_plots

OUT_DIR = "ttt_logging/csv"
WINDOW = int(os.environ.get("TTT_WINDOW", "51"))

# Per experiment: tag prefix + figbase. gXX tags are <prefix>_gNN; g00 restricts to X
# (no Z present at growth=0), g50/g100 restrict to Z. Override via CLI:
#     python plot_z332_z_only.py <tag_prefix> <figbase>
# e.g.  python plot_z332_z_only.py z332_persample_sent figs/ttt/z332/ramp_persample/sentences
DEFAULT_PREFIX = "z332_ramphlr_sent"
DEFAULT_FIGBASE = "figs/ttt/z332/ramp_highlr/sentences"

# The batch_size x lr grid actually swept by run_z332_testing_sweep.sh / rerun_z332_testing_
# smallbatch.sh (batch_size=1 abstract is dropped -- see rerun script header). Tag/figdir
# naming here must match those scripts' `tag`/`figdir` construction exactly.
Z332_TESTING_GRID = {
    "sentence": {"abstract": 0, "batches": [4, 16, 64],
                 "lrs": ["1e-5", "1e-3", "1e-2", "1e-1"]},
    "abstract": {"abstract": 1, "batches": [2, 4],
                 "lrs": ["1e-2", "1e-3", "1e-4"]},
}


def build_jobs(prefix, figbase):
    return [
        (f"{prefix}_g00",  f"{figbase}/g00",  "X"),
        (f"{prefix}_g50",  f"{figbase}/g50",  "Z"),
        (f"{prefix}_g100", f"{figbase}/g100", "Z"),
    ]


def build_jobs_z332_testing():
    """Same (tag, figdir, source) triples as build_jobs(), but for every
    granularity x batch_size x lr x gXX combo in the z332_testing sweep."""
    jobs = []
    for gran, cfg in Z332_TESTING_GRID.items():
        for bs in cfg["batches"]:
            for lr in cfg["lrs"]:
                figbase = f"figs/ttt/z332_testing/{gran}/batch_{bs}_lr_{lr}"
                for gtag, source in (("g00", "X"), ("g50", "Z"), ("g100", "Z")):
                    tag = f"z332test_{gran}_b{bs}_lr{lr}_{gtag}"
                    jobs.append((tag, f"{figbase}/{gtag}", source))
    return jobs


def build_set1(tag, base_figdir, source):
    samples_csv = f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_samples_{tag}.csv"
    if not os.path.exists(samples_csv):
        print(f"  [missing] {samples_csv} (job not finished?)")
        return
    df = pd.read_csv(samples_csv)
    if "ai_source" not in df.columns:
        print(f"  [skip] {tag}: no ai_source column (not a gradual run)")
        return
    keep = (df["label"] == "human") | ((df["label"] == "ai") & (df["ai_source"] == source))
    sub = df[keep].copy()
    n_ai = int((sub["label"] == "ai").sum())
    if n_ai == 0:
        print(f"  [skip] {tag}: no AI rows with source={source}")
        return

    outdir = os.path.join(base_figdir, "z332_only")
    os.makedirs(outdir, exist_ok=True)
    filt_csv = f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_samples_{tag}_z332only.csv"
    sub.to_csv(filt_csv, index=False)

    src_label = "rewrite_Z_332" if source == "Z" else "rewrite_X"
    suffix = f"  (sentence, {src_label} only)"
    make_curves(filt_csv, outdir, window=WINDOW, title_suffix=suffix)
    make_eda_pred_hist(filt_csv, outdir,
                       modes=tuple(sorted(sub["strategy"].unique())), title_suffix=suffix)
    make_ramp_plots(filt_csv, outdir, title_suffix=suffix)   # no-op for the X-only g00 set
    print(f"  SET1 {tag}: {n_ai} {source}-AI rows -> {outdir}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "z332_testing":
        print("SET1 (rewrite_Z_332 only) for the full z332_testing grid "
              "-> figs/ttt/z332_testing/**/gXX/z332_only/")
        for tag, figdir, source in build_jobs_z332_testing():
            build_set1(tag, figdir, source)
    else:
        prefix = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PREFIX
        figbase = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FIGBASE
        print(f"SET1 (rewrite_Z_332 only) for prefix={prefix} -> {figbase}/gXX/z332_only/")
        for tag, figdir, source in build_jobs(prefix, figbase):
            build_set1(tag, figdir, source)
    print("done. SET2 (entire batch / all AI) is in each g-dir's root, produced by the sbatch merge.")
