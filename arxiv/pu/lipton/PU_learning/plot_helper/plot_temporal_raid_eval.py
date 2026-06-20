"""
Plots for logging_accuracy_temporal_raid_eval.csv.

One bar chart per unique value of test_col.
Each bar represents the metric for a unique test_val.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", weight="bold", size=20)

INPUT_CSV = "../logging_accuracy_temporal_raid_eval.csv"
OUTPUT_FOLDER = "logging_accuracy_temporal_raid_eval"

METRIC = "auc"
CI = "0.95"

# Display names for test_col groups
TEST_COL_DISPLAY = {
    "attack": "Attack Type",
    "domain": "Domain",
    "repetition_penalty": "Repetition Penalty",
    "decoding": "Decoding Strategy",
}

BAR_COLOR = "steelblue"


def fmt_label(s):
    return s.replace("_", "\n")


def plot_test_col(df, test_col, ax, title=True):
    sub = df[df["test_col"] == test_col].copy()
    test_vals = list(sub["test_val"])
    n = len(test_vals)

    bar_width = 0.6
    x = np.arange(n)

    heights, errs_lo, errs_hi = [], [], []
    for _, row in sub.iterrows():
        val = row[METRIC]
        lo = val - row[f"{METRIC}_l_{CI}"]
        hi = row[f"{METRIC}_u_{CI}"] - val
        heights.append(val)
        errs_lo.append(lo)
        errs_hi.append(hi)

    ax.bar(x, heights, width=bar_width, color=BAR_COLOR, alpha=0.85, zorder=3)
    ax.errorbar(x, heights, yerr=[errs_lo, errs_hi],
                fmt="none", ecolor="black", elinewidth=1.5, capsize=4, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_label(v) for v in test_vals],
                       fontsize=14, fontweight="bold")
    ax.set_ylabel("AUC", fontsize=18, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(TEST_COL_DISPLAY.get(test_col, test_col),
                     fontsize=20, fontweight="bold")


def main():
    df = pd.read_csv(INPUT_CSV)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    test_cols = list(df["test_col"].unique())

    # Save one PDF per test_col
    for tc in test_cols:
        n_vals = df[df["test_col"] == tc]["test_val"].nunique()
        width = max(6, 1.0 * n_vals)
        fig, ax = plt.subplots(figsize=(width, 5))
        plot_test_col(df, tc, ax, title=True)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_FOLDER, f"bar_{tc}.pdf")
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()
        plt.close(fig)
        print(f"Saved {save_path}")

    # Also save a combined grid figure
    n = len(test_cols)
    n_cols_grid = min(2, n)
    n_rows_grid = int(np.ceil(n / n_cols_grid))

    # Determine per-column widths for the grid (wider for more test_vals)
    col_widths = []
    for i in range(n_cols_grid):
        idxs = range(i, n, n_cols_grid)
        max_vals = max(df[df["test_col"] == test_cols[idx]]["test_val"].nunique()
                       for idx in idxs if idx < n)
        col_widths.append(max(5, 1.0 * max_vals))

    fig, axes = plt.subplots(
        n_rows_grid, n_cols_grid,
        figsize=(sum(col_widths), 5 * n_rows_grid),
        gridspec_kw={"width_ratios": col_widths},
    )
    axes_flat = np.array(axes).flatten()

    for idx, tc in enumerate(test_cols):
        plot_test_col(df, tc, axes_flat[idx], title=True)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, "bar_all_test_cols.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
