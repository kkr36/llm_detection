"""
Same plots as plot_optimization_xz.py, but with a merged data source:

  - Every row from logging_accuracy_xz_v2.csv is used as-is.
  - Rows from logging_accuracy_xz.csv are kept only when their
    {learning_method, train_llm, eval_llm, train_alpha} key does NOT appear
    in the v2 CSV (i.e. v2 substitutes wherever it has data; the original
    CSV backfills everything else, e.g. pangram rows).

Output goes to plot_helper/logging_accuracy_xz_naive_mirror/.

Run from the plot_helper directory, same as plot_optimization_xz.py:
    cd plot_helper && python plot_optimization_xz_naive_mirror.py
"""

import pandas as pd

# Reuse all the plotting logic + transforms from the original script.
import plot_optimization_xz as P

# Substitution key: a v2 row replaces base rows with the same tuple.
MERGE_KEY = ["learning_method", "train_llm", "eval_llm", "train_alpha"]

BASE_CSV = "../logging_accuracy_xz.csv"
V2_CSV   = "../logging_accuracy_xz_v2.csv"
PNU_CSV  = "../logging_accuracy_xz_PNU.csv"

OUTPUT_FOLDER = "logging_accuracy_xz_naive_mirror"

# Full metric set behind the figures in logging_accuracy_xz_paper/.
PAPER_METRICS = ["auc", "accuracy", "pos_prob", "neg_prob",
                 "bce", "bbe", "plugin-int", "tpr", "tnr"]


def run_grid(df, metrics, title, pnu_df=None):
    """Drive the grid figures via the `_part` helper directly, avoiding the
    stray pdb.set_trace() in the non-PNU grid wrapper. Splits into exactly two
    near-even pages (e.g. 9 metrics -> 5 + 4), matching the two grid files in
    logging_accuracy_xz_paper/ while still covering every metric."""
    import math
    segments = 2
    per_segment = math.ceil(len(metrics) / segments)
    segs = [metrics[per_segment * i:per_segment * (i + 1)]
            for i in range(segments)]
    for i, seg in enumerate(segs):
        if pnu_df is None:
            P._make_xz_barplot_fig_ab_grid_part(df, seg, str(i), title=title)
        else:
            P._make_xz_barplot_fig_ab_grid_pnu_part(df, pnu_df, seg, str(i), title=title)


def merge_v2_over_base(base: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    """All v2 rows, plus base rows whose MERGE_KEY is absent from v2.

    Column sets differ between the two files (v2 has iteration/eval_parquet;
    base has pangram_* columns); pd.concat aligns on the union of columns and
    fills the gaps with NaN, which is exactly what the downstream filters
    (e.g. pangram_score_type) expect.
    """
    v2_keys = set(map(tuple, v2[MERGE_KEY].itertuples(index=False, name=None)))
    keep_mask = [
        key not in v2_keys
        for key in base[MERGE_KEY].itertuples(index=False, name=None)
    ]
    base_kept = base[keep_mask]

    n_dropped = len(base) - len(base_kept)
    print(f"[merge] v2 rows: {len(v2)}")
    print(f"[merge] base rows: {len(base)} "
          f"({n_dropped} overridden by v2, {len(base_kept)} kept)")

    merged = pd.concat([v2, base_kept], ignore_index=True)
    print(f"[merge] merged rows: {len(merged)}")
    return merged


if __name__ == "__main__":
    # Redirect all saved figures into the new output folder.
    P.output_folder = OUTPUT_FOLDER

    base = pd.read_csv(BASE_CSV)
    v2   = pd.read_csv(V2_CSV)
    data = merge_v2_over_base(base, v2)

    # Same post-processing pipeline as the original __main__.
    data = P.swap_pangram_tpr_tnr(data)
    data = P.add_accuracy_cols(data)
    data = P.reverse_bias(data)
    data = P.reverse_plugin(data)

    data_pnu = pd.read_csv(PNU_CSV)
    data_pnu = P.add_accuracy_cols(data_pnu)
    data_pnu = P.reverse_bias(data_pnu)
    data_pnu = P.reverse_plugin(data_pnu)

    for use_title in [False, True][1:]:
        # xz_barplot_fig_ab_pnu_{metric}.pdf  (one per metric)
        P.make_xz_barplot_fig_ab_pnu(data, data_pnu, PAPER_METRICS, title=use_title)
        # xz_barplot_fig_ab_tnr.pdf
        P.make_xz_barplot_fig_ab(data, ["tnr"], title=use_title)
        # xz_barplot_fig_ab_grid_{i}.pdf
        run_grid(data, PAPER_METRICS, title=use_title)
        # xz_barplot_fig_ab_grid_pnu_{i}.pdf
        run_grid(data, PAPER_METRICS, title=use_title, pnu_df=data_pnu)
