"""
plot_judge_scores.py

Loads the parquet produced by llm_judge.py and saves a single PDF with two figures:

  Page 1 — Scatter correlation
      2 rows (hallucination / omission) × N rewrite columns
      Each cell: scatter of mean_model_score vs judge score + regression line + r/p

  Page 2 — Binned summary
      Same 2×N grid.
      mean_model_score is binned into 5 equal-width bins [0, 1].
      Each cell: bar chart showing mean ± std of the judge score per bin.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────

# RESULTS_PARQUET = (
#     "/share/garg/arxiv_kaggle/multillm/data_raw/"
#     "faithfulness_scores_2020_xyz_preds.parquet"
# )
RESULTS_PARQUET = "/share/garg/arxiv_kaggle/multillm/data_raw/faithfulness_scores_2020_xyz_preds_2.parquet"
OUTPUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_plots.pdf")
HISTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hists")

N_BINS = 5
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)   # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]

SCORE_PAIRS = [
    ("hallucination_score", "mean_model_score", "Hallucination score"),
    ("omission_score",      "mean_model_score", "Omission score"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_scatter_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    """Reproduce the correlation_model_vs_judge scatter plot."""
    n_rows = len(SCORE_PAIRS)
    n_cols = len(rewrite_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row_i, (x_col, y_col, y_label) in enumerate(SCORE_PAIRS):
        for col_i, rcol in enumerate(rewrite_cols):
            ax = axes[row_i][col_i]
            sub = results_df[results_df["rewrite_col"] == rcol][[x_col, y_col]].dropna()

            ax.scatter(sub[x_col], sub[y_col], alpha=0.5, s=20)

            if len(sub) >= 3:
                r, p = stats.pearsonr(sub[x_col], sub[y_col])
                m, b = np.polyfit(sub[x_col], sub[y_col], 1)
                xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
                ax.plot(xs, m * xs + b, color="red", linewidth=1.2)
                ax.set_title(f"{rcol}\nr={r:.2f}, p={p:.3f}", fontsize=9)
            else:
                ax.set_title(rcol, fontsize=9)

            ax.set_xlabel(y_label, fontsize=8)
            ax.set_ylabel("Model score (mean eligible folds)", fontsize=8)
            ax.autoscale()
            ax.margins(0.05)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Correlation: model detection score vs LLM judge scores\n(per rewrite type)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig


def make_binned_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    """Bar chart of mean ± std of judge scores, binned by model score."""
    n_rows = len(SCORE_PAIRS)
    n_cols = len(rewrite_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row_i, (x_col, y_col, y_label) in enumerate(SCORE_PAIRS):
        for col_i, rcol in enumerate(rewrite_cols):
            ax = axes[row_i][col_i]
            sub = results_df[results_df["rewrite_col"] == rcol][[x_col, y_col]].dropna()

            bin_labels = pd.cut(
                sub[x_col],
                bins=BIN_EDGES,
                labels=range(N_BINS),
                include_lowest=True,
            )
            grouped = sub[y_col].groupby(bin_labels)
            bin_means = grouped.mean().reindex(range(N_BINS))
            bin_stds  = grouped.std().reindex(range(N_BINS))
            bin_ns    = grouped.count().reindex(range(N_BINS))

            # bar chart
            bars = ax.bar(
                BIN_CENTERS,
                bin_means,
                width=BIN_WIDTH * 0.7,
                yerr=bin_stds,
                capsize=4,
                color="steelblue",
                alpha=0.7,
                error_kw={"elinewidth": 1.2},
            )

            # annotate n per bin — just above the bar
            y_top = ax.get_ylim()[1]
            # for center, mean_val, n in zip(BIN_CENTERS, bin_means, bin_ns):
            #     if not np.isnan(n) and n > 0 and not np.isnan(mean_val):
            #         ax.text(center, mean_val + 0.02 * y_top, f"n={int(n)}",
            #                 ha="center", va="bottom", fontsize=6, color="gray")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.1)
            ax.set_xticks(BIN_CENTERS)
            ax.set_xticklabels(
                [f"{BIN_EDGES[i]:.1f}–{BIN_EDGES[i+1]:.1f}" for i in range(N_BINS)],
                fontsize=6, rotation=30,
            )
            ax.set_xlabel(y_label + " bin", fontsize=8)
            ax.set_ylabel("Mean model score", fontsize=8)
            ax.set_title(rcol, fontsize=9)
            ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        "Binned model score vs LLM judge scores (mean ± std per bin)\n(per rewrite type)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig


def save_hist_figs(results_df: pd.DataFrame, rewrite_cols: list[str]) -> None:
    """Save per-prompt histograms of hallucination and omission scores to HISTS_DIR."""
    os.makedirs(HISTS_DIR, exist_ok=True)

    score_cols = [
        ("hallucination_score", "Hallucination score"),
        ("omission_score",      "Omission / dropping score"),
    ]

    for rcol in rewrite_cols:
        sub = results_df[results_df["rewrite_col"] == rcol]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, (col, label) in zip(axes, score_cols):
            vals = sub[col].dropna()
            ax.hist(vals, bins=20, range=(0, 1), color="steelblue", alpha=0.75, edgecolor="white")
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_title(f"{label}\nn={len(vals)}, mean={vals.mean():.3f}", fontsize=9)
            ax.tick_params(labelsize=8)

        safe_name = rcol.replace("/", "_").replace(" ", "_")
        fig.suptitle(f"Score distributions — {rcol}", fontsize=11)
        fig.tight_layout()
        out_path = os.path.join(HISTS_DIR, f"{safe_name}.pdf")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"  Saved hist → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {RESULTS_PARQUET} …")
    results_df = pd.read_parquet(RESULTS_PARQUET)

    rewrite_cols = sorted(results_df["rewrite_col"].dropna().unique().tolist())
    print(f"Rewrite columns: {rewrite_cols}")

    with PdfPages(OUTPUT_PDF) as pdf:
        fig1 = make_scatter_fig(results_df, rewrite_cols)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = make_binned_fig(results_df, rewrite_cols)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    print(f"Saved → {OUTPUT_PDF}")

    print(f"Saving per-prompt histograms to {HISTS_DIR} …")
    save_hist_figs(results_df, rewrite_cols)
    print("Done.")
