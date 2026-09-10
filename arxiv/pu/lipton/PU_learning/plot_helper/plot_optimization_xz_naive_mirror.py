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

import os

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Reuse all the plotting logic + transforms from the original script.
import plot_optimization_xz as P

# Substitution key: a v2 row replaces base rows with the same tuple.
MERGE_KEY = ["learning_method", "train_llm", "eval_llm", "train_alpha"]

BASE_CSV = "../logging_accuracy_xz.csv"
V2_CSV   = "../logging_accuracy_xz_v2.csv"
PNU_CSV  = "../logging_accuracy_xz_PNU.csv"

OUTPUT_FOLDER = "logging_accuracy_xz_naive_mirror"
PAPER_OUTPUT_FOLDER = f"{OUTPUT_FOLDER}/paper"

# Full metric set behind the figures in logging_accuracy_xz_paper/.
PAPER_METRICS = ["auc", "accuracy", "pos_prob", "neg_prob",
                 "bce", "bbe", "plugin-int", "tpr", "tnr"]

# Strategy-Z iteration 0 replaces the old rewrite_Z comparison at t=1. The
# later mirror-game iterations continue to use the regenerated v2 rows.
STRATEGY0_PU_TRAJECTORY = [
    ("TEDn", "X",    "rewrite_X"),
    ("TEDn", "xz0",  "rewrite_strategy_Z_0"),
    ("TEDn", "xzz",  "rewrite_Z_1_PU"),
    ("TEDn", "xzzz", "rewrite_Z_2_PU"),
]

STRATEGY0_PN_TRAJECTORY = [
    ("PN", "X",   "rewrite_X"),
    ("PN", "X",   "rewrite_strategy_Z_0"),
    ("PN", "xz",  "rewrite_Z_1_PN"),
    ("PN", "xzz", "rewrite_Z_2_PN"),
]

PANGRAM_332_METHOD = "pangram_3.3.2"
PANGRAM_332_COLOR = "deepskyblue"


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

def swap_method_tpr_tnr(df, method, ci_level=0.95):
    """Apply Pangram's label-orientation correction to one named version."""
    df = df.copy()
    ci = str(ci_level)
    mask = df["learning_method"] == method
    for tpr_col, tnr_col in zip(
        ["tpr", f"tpr_l_{ci}", f"tpr_u_{ci}"],
        ["tnr", f"tnr_l_{ci}", f"tnr_u_{ci}"],
    ):
        if tpr_col in df.columns and tnr_col in df.columns:
            df.loc[mask, [tpr_col, tnr_col]] = df.loc[
                mask, [tnr_col, tpr_col]
            ].values
    return df


def get_pangram_version_vals(
    df,
    method,
    eval_llm_col,
    point_col,
    lower_col,
    upper_col,
    score_type=P.PANGRAM_SCORE_TYPE,
):
    """Return one Pangram version's point and 95% interval for an eval column."""
    row = df[
        (df["learning_method"] == method)
        & (df["eval_llm"] == eval_llm_col)
        & (df["pangram_score_type"] == score_type)
    ]
    if row.empty or point_col not in row.columns:
        return np.nan, np.nan, np.nan

    point = row[point_col].mean()
    lower = row[lower_col].mean() if lower_col in row.columns else np.nan
    upper = row[upper_col].mean() if upper_col in row.columns else np.nan

    # Match the existing Pangram transformations in plot_optimization_xz.py.
    if (
        "pos_prob" in point_col
        or "neg_prob" in point_col
        or "plugin-int" in point_col
    ):
        point, lower, upper = 1 - point, 1 - upper, 1 - lower
    return point, lower, upper


def _pair_eval_values(first, second):
    """Turn two (point, lower, upper) tuples into three two-element arrays."""
    return tuple(np.array([first[i], second[i]]) for i in range(3))


def _strategy0_plot_values(df, df_pnu, metric):
    """Collect every value used by one row of the strategy-Z-0 paper figure."""
    point_col, lower_col, upper_col, _ = P.resolve_cols(metric)

    values = {
        "pu_a": P._get_trajectory_vals(
            df,
            STRATEGY0_PU_TRAJECTORY[:2],
            point_col,
            lower_col,
            upper_col,
        ),
        "pn_a": P._get_trajectory_vals(
            df,
            STRATEGY0_PN_TRAJECTORY[:2],
            point_col,
            lower_col,
            upper_col,
        ),
        "pu_b": P._get_trajectory_vals(
            df,
            STRATEGY0_PU_TRAJECTORY[1:],
            point_col,
            lower_col,
            upper_col,
        ),
        "pn_b": P._get_trajectory_vals(
            df,
            STRATEGY0_PN_TRAJECTORY[1:],
            point_col,
            lower_col,
            upper_col,
        ),
        "pangram_332": _pair_eval_values(
            get_pangram_version_vals(
                df,
                PANGRAM_332_METHOD,
                "rewrite_X",
                point_col,
                lower_col,
                upper_col,
            ),
            get_pangram_version_vals(
                df,
                PANGRAM_332_METHOD,
                "rewrite_Z_332",
                point_col,
                lower_col,
                upper_col,
            ),
        ),
        "pnu": _pair_eval_values(
            P._get_pnu_vals(
                df_pnu, "rewrite_X", point_col, lower_col, upper_col
            ),
            P._get_pnu_vals(
                df_pnu, "rewrite_Z", point_col, lower_col, upper_col
            ),
        ),
    }

    if metric in ("bbe", "plugin-int"):
        values = {
            name: tuple(component - 0.5 for component in components)
            for name, components in values.items()
        }
    return values


def _paper_legend_handles():
    return [
        mpatches.Patch(color="purple", label="PU + TTA"),
        mpatches.Patch(color=P.PNU_COLOR, label="PNU + TTA"),
        mpatches.Patch(color="orange", label="Supervised"),
        mpatches.Patch(color=PANGRAM_332_COLOR, label="Pangram 3.3.2"),
    ]


def _draw_strategy0_fig_ab_axes(ax_a, ax_b, df, df_pnu, metric, grid=False):
    """Draw one metric's two panels on caller-provided axes."""
    values = _strategy0_plot_values(df, df_pnu, metric)
    ts_labels = ["Naive\nprompt", "Adversarial\nhumanizing\nprompt"]
    iter_labels = ["Iteration\n1", "Iteration\n2", "Iteration\n3"]

    bar_width_a = 0.2
    x_a = np.arange(len(ts_labels))
    left_series = [
        (values["pu_a"], "purple"),
        (values["pnu"], P.PNU_COLOR),
        (values["pn_a"], "orange"),
        (values["pangram_332"], PANGRAM_332_COLOR),
    ]
    for offset, (series, color) in zip(
        np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width_a, left_series
    ):
        points, lowers, uppers = series
        ax_a.bar(
            x_a + offset,
            points,
            bar_width_a,
            color=color,
            yerr=P._safe_yerr(points, lowers, uppers),
            capsize=4 if grid else 5,
            error_kw={"linewidth": 1.5 if grid else 2},
        )
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(ts_labels, fontsize=20 if grid else 22)
    ax_a.set_xlabel(
        "Single-shot adversarial prompt", fontsize=20 if grid else 25
    )
    ax_a.set_ylabel(
        P.name_to_name.get(metric, metric), fontsize=20 if grid else 35
    )

    bar_width_b = 0.35 if grid else 0.25
    x_b = np.arange(len(iter_labels))
    pu_points, pu_lowers, pu_uppers = values["pu_b"]
    pn_points, pn_lowers, pn_uppers = values["pn_b"]
    ax_b.bar(
        x_b - bar_width_b / 2,
        pu_points,
        bar_width_b,
        color="purple",
        yerr=P._safe_yerr(pu_points, pu_lowers, pu_uppers),
        capsize=4 if grid else 5,
        error_kw={"linewidth": 1.5 if grid else 2},
    )
    ax_b.bar(
        x_b + bar_width_b / 2,
        pn_points,
        bar_width_b,
        color="orange",
        yerr=P._safe_yerr(pn_points, pn_lowers, pn_uppers),
        capsize=4 if grid else 5,
        error_kw={"linewidth": 1.5 if grid else 2},
    )
    ax_b.axhline(
        y=np.nanmean(pu_points),
        color="purple",
        linewidth=2 if grid else 2.5,
        linestyle="--",
        alpha=0.8,
    )
    ax_b.axhline(
        y=np.nanmean(pn_points),
        color="orange",
        linewidth=2 if grid else 2.5,
        linestyle="--",
        alpha=0.8,
    )
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(iter_labels, fontsize=20 if grid else 22)
    ax_b.set_xlabel(
        "Iterated detector-evader game",
        fontsize=20 if grid else 25,
        labelpad=25 if grid else 30,
    )

    if grid:
        title = P.name_to_name.get(metric, metric)
        ax_a.set_title(title, fontsize=20, fontweight="bold")
        ax_b.set_title(title, fontsize=20, fontweight="bold")


def make_strategy0_paper_barplot(df, df_pnu, metrics):
    """Write individual strategy-Z-0 figures with Pangram 3.3.2."""
    os.makedirs(PAPER_OUTPUT_FOLDER, exist_ok=True)
    for metric in metrics:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(22, 7))
        _draw_strategy0_fig_ab_axes(ax_a, ax_b, df, df_pnu, metric)
        fig.legend(
            handles=_paper_legend_handles(),
            loc="upper center",
            ncol=4,
            fontsize=21,
            bbox_to_anchor=(0.5, 1.06),
            frameon=False,
        )
        plt.tight_layout()
        plt.savefig(
            f"{PAPER_OUTPUT_FOLDER}/xz_barplot_fig_ab_pnu_{metric}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def _make_strategy0_paper_grid_part(
    df, df_pnu, metrics, filename_suffix
):
    """Write one vertically stacked grid page for up to four metrics."""
    with matplotlib.rc_context({"font.size": 14, "font.weight": "bold"}):
        fig, axes = plt.subplots(
            len(metrics),
            2,
            figsize=(22, 7 * len(metrics)),
            squeeze=False,
        )
        for row_i, metric in enumerate(metrics):
            _draw_strategy0_fig_ab_axes(
                axes[row_i, 0],
                axes[row_i, 1],
                df,
                df_pnu,
                metric,
                grid=True,
            )
        fig.legend(
            handles=_paper_legend_handles(),
            loc="upper center",
            ncol=4,
            fontsize=18,
            bbox_to_anchor=(0.5, 0.995),
            frameon=False,
        )
        top = 0.90 if len(metrics) == 1 else 0.97
        plt.tight_layout(rect=(0, 0, 1, top))
        plt.savefig(
            (
                f"{PAPER_OUTPUT_FOLDER}/"
                f"xz_barplot_fig_ab_grid_pnu_{filename_suffix}.pdf"
            ),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def make_strategy0_paper_grids(
    df, df_pnu, metrics, metrics_per_file=4
):
    """Split metrics into consecutive groups of four and write each grid."""
    os.makedirs(PAPER_OUTPUT_FOLDER, exist_ok=True)
    for page, start in enumerate(range(0, len(metrics), metrics_per_file)):
        _make_strategy0_paper_grid_part(
            df,
            df_pnu,
            metrics[start:start + metrics_per_file],
            str(page),
        )



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

    paper_data = swap_method_tpr_tnr(data, PANGRAM_332_METHOD)
    paper_data = P.add_accuracy_cols(paper_data)

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

    # Paper variants: one AI-recall figure and grids of four metrics per PDF.
    make_strategy0_paper_barplot(paper_data, data_pnu, ["tnr"])
    make_strategy0_paper_grids(paper_data, data_pnu, PAPER_METRICS)
