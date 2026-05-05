"""
plot_judge_scores_grid.py

Produces two new publication-ready PDFs from the judge-score parquet:

  judge_hists_grid.pdf   — all histograms in one figure (1 prompt per row, 2 score cols)
  judge_scatter_grid.pdf — all scatter plots in a 2-column grid (bigger dots, thicker lines)

Prompt display names (rewrite_col values have a "rewrite_" prefix and uppercase X/Z):
  rewrite_X              → "Naive Prompt"
  rewrite_Z              → "Adversarial Humanizing Prompt"
  rewrite_Z_{t}_{method} → "Iteration {t}, Supervised"  (method == PN)
                         → "Iteration {t}, PU + TTA"    (otherwise)
"""

import math
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────

RESULTS_PARQUET   = "/share/garg/arxiv_kaggle/multillm/data_raw/faithfulness_scores_2020_xyz_preds_2.parquet"
_HERE             = os.path.dirname(os.path.abspath(__file__))
HISTS_GRID_PDF    = os.path.join(_HERE, "judge_hists_grid.pdf")
SCATTER_GRID_PDF  = os.path.join(_HERE, "judge_scatter_grid.pdf")

SCORE_PAIRS = [
    ("hallucination_score", "mean_model_score", "Hallucination Score"),
    ("omission_score",      "mean_model_score", "Omission Score"),
]

SCORE_COLS = [
    ("hallucination_score", "Hallucination Score"),
    ("omission_score",      "Omission / Dropping Score"),
]


# ── Prompt renaming ────────────────────────────────────────────────────────────

def rename_prompt(col: str) -> str:
    # Strip "rewrite_" prefix, then normalise to uppercase for matching
    name = col[len("rewrite_"):] if col.lower().startswith("rewrite_") else col
    name_up = name.upper()

    if name_up == "X":
        return "Naive Prompt"
    if name_up == "Z":
        return "Adversarial Humanizing Prompt"

    # Pattern: Z_{t}_{method}  e.g. Z_1_PN, Z_2_PU
    parts = name.split("_", 2)
    if len(parts) == 3 and parts[0].upper() == "Z":
        try:
            t = int(parts[1])
        except ValueError:
            return col
        method = parts[2].upper()
        label = "Supervised" if method == "PN" else "PU + TTA"
        return f"Iteration {t}, {label}"
    return col


# ── Combined histogram grid ────────────────────────────────────────────────────

def make_combined_hist_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    """One row per prompt, two columns (hallucination / omission)."""
    n_rows = len(rewrite_cols)
    n_cols = len(SCORE_COLS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for row_i, rcol in enumerate(rewrite_cols):
        sub = results_df[results_df["rewrite_col"] == rcol]
        display_name = rename_prompt(rcol)

        is_bottom_row = (row_i == n_rows - 1)
        for col_i, (col, label) in enumerate(SCORE_COLS):
            ax = axes[row_i][col_i]
            vals = sub[col].dropna()
            ax.hist(vals, bins=20, range=(0, 1),
                    color="steelblue", alpha=0.75, edgecolor="white")
            if is_bottom_row:
                ax.set_xlabel(label, fontsize=18, fontweight='bold')
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            ax.set_ylabel("Count", fontsize=18)
            ax.set_xlim(0, 1)
            ax.set_title(
                f"{display_name}\nn={len(vals)}, mean={vals.mean():.3f}",
                fontsize=18,
            )
            ax.tick_params(labelsize=15)

    # fig.suptitle("Score Distributions", fontsize=24, y=1.01)
    fig.tight_layout()
    return fig


# ── Combined scatter grid (2 cols) ─────────────────────────────────────────────

def make_combined_scatter_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    """All (rewrite_col × score_pair) scatter plots in a 2-column grid."""
    n_grid_cols = 2
    all_cells = [
        (rcol, sp)
        for rcol in rewrite_cols
        for sp in SCORE_PAIRS
    ]
    n_cells = len(all_cells)
    n_grid_rows = math.ceil(n_cells / n_grid_cols)

    fig, axes = plt.subplots(
        n_grid_rows, n_grid_cols,
        figsize=(7 * n_grid_cols, 6 * n_grid_rows),
        squeeze=False,
    )

    for idx, (rcol, (x_col, y_col, y_label)) in enumerate(all_cells):
        row_i = idx // n_grid_cols
        col_i = idx % n_grid_cols
        ax = axes[row_i][col_i]

        sub = results_df[results_df["rewrite_col"] == rcol][[x_col, y_col]].dropna()
        display_name = rename_prompt(rcol)

        ax.scatter(sub[x_col], sub[y_col], alpha=0.5, s=80)

        if len(sub) >= 3:
            r, p = stats.pearsonr(sub[x_col], sub[y_col])
            m, b = np.polyfit(sub[x_col], sub[y_col], 1)
            xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
            ax.plot(xs, m * xs + b, color="red", linewidth=3.0)
            ax.set_title(f"{display_name}\nr={r:.2f}, p={p:.3f}", fontsize=18)
        else:
            ax.set_title(display_name, fontsize=18)

        # Only show x-axis label on the last row of each column
        last_row_for_col = n_grid_rows - 1 if (n_cells % n_grid_cols == 0 or col_i < n_cells % n_grid_cols) else n_grid_rows - 2
        if row_i == last_row_for_col:
            ax.set_xlabel(y_label, fontsize=16, fontweight='bold')
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        ax.set_ylabel("Model score (mean eligible folds)", fontsize=16)
        ax.autoscale()
        ax.margins(0.05)
        ax.tick_params(labelsize=14)

    for idx in range(n_cells, n_grid_rows * n_grid_cols):
        axes[idx // n_grid_cols][idx % n_grid_cols].set_visible(False)

    # fig.suptitle(
    #     "Correlation: Model Detection Score vs LLM Judge Scores",
    #     fontsize=24, y=1.01,
    # )
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {RESULTS_PARQUET} …")
    results_df = pd.read_parquet(RESULTS_PARQUET)

    rewrite_cols = sorted(results_df["rewrite_col"].dropna().unique().tolist())
    print(f"Rewrite columns: {rewrite_cols}")
    print(f"Display names:   {[rename_prompt(c) for c in rewrite_cols]}")

    fig_hists = make_combined_hist_fig(results_df, rewrite_cols)
    fig_hists.savefig(HISTS_GRID_PDF, dpi=150, bbox_inches="tight", format="pdf")
    plt.close(fig_hists)
    print(f"Saved → {HISTS_GRID_PDF}")

    fig_scatter = make_combined_scatter_fig(results_df, rewrite_cols)
    fig_scatter.savefig(SCATTER_GRID_PDF, dpi=150, bbox_inches="tight", format="pdf")
    plt.close(fig_scatter)
    print(f"Saved → {SCATTER_GRID_PDF}")

    print("Done.")
