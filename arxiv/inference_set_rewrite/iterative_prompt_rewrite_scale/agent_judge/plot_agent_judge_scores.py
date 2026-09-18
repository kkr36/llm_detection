"""
Plot and summarize coding-agent faithfulness judgments.

This script is intentionally self-contained within agent_judge: it reads the
agent-produced parquet and does not merge scores from older LLM-judge runs,
which may have been produced from different source texts.

Outputs, by default under plots/:

  agent_judge_plots.pdf             Multi-page PDF with summary, grids, scatter.
  agent_judge_hists_grid.pdf        Publication-style histogram grid.
  agent_judge_score_pair_grid.pdf   Hallucination-vs-omission scatter grid.
  hists/*.pdf                       Per-rewrite histogram PDFs.
  agent_judge_score_summary.csv     Per-rewrite aggregate statistics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_PARQUET = HERE / "faithfulness_scores_agent.parquet"
DEFAULT_OUT_DIR = HERE / "plots"

SCORE_COLS = [
    ("hallucination_score", "Hallucination Score"),
    ("omission_score", "Omission Score"),
]

MODEL_SCORE_PAIRS = [
    ("hallucination_score", "mean_model_score", "Hallucination Score"),
    ("omission_score", "mean_model_score", "Omission Score"),
]


def rename_prompt(col: str) -> str:
    """Return display names matching the project plotting convention."""
    name = col[len("rewrite_") :] if col.lower().startswith("rewrite_") else col
    name_up = name.upper()

    if name_up == "X":
        return "Naive Prompt"
    if name_up == "Z":
        return "Adversarial Humanizing Prompt"
    if name_up == "Z_332":
        return "Adversarial Humanizing Prompt"

    parts = name.split("_", 2)
    if len(parts) == 3 and parts[0].upper() == "Z":
        try:
            iteration = int(parts[1])
        except ValueError:
            return col
        method = parts[2].upper()
        label = "Supervised" if method == "PN" else "PU + TTA"
        return f"Iteration {iteration}, {label}"

    return col


def load_results(path: Path) -> pd.DataFrame:
    results_df = pd.read_parquet(path)
    required = {"rewrite_col", "hallucination_score", "omission_score"}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return results_df


def ordered_rewrite_cols(results_df: pd.DataFrame) -> list[str]:
    preferred = [
        "rewrite_X",
        "rewrite_Z_332",
        "rewrite_Z_1_PU",
        "rewrite_Z_1_PN",
        "rewrite_Z_2_PU",
        "rewrite_Z_2_PN",
    ]
    present = set(results_df["rewrite_col"].dropna().unique())
    ordered = [col for col in preferred if col in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def make_summary_table(results_df: pd.DataFrame, rewrite_cols: list[str]) -> pd.DataFrame:
    rows = []
    for rcol in rewrite_cols:
        sub = results_df[results_df["rewrite_col"] == rcol]
        hall = sub["hallucination_score"].dropna()
        om = sub["omission_score"].dropna()
        both = sub[["hallucination_score", "omission_score"]].dropna()

        if len(both) >= 3 and both["hallucination_score"].nunique() > 1 and both["omission_score"].nunique() > 1:
            corr_r, corr_p = stats.pearsonr(both["hallucination_score"], both["omission_score"])
        else:
            corr_r, corr_p = np.nan, np.nan

        rows.append(
            {
                "rewrite_col": rcol,
                "display_name": rename_prompt(rcol),
                "n": len(sub),
                "hallucination_mean": hall.mean(),
                "hallucination_std": hall.std(),
                "hallucination_median": hall.median(),
                "hallucination_min": hall.min(),
                "hallucination_max": hall.max(),
                "omission_mean": om.mean(),
                "omission_std": om.std(),
                "omission_median": om.median(),
                "omission_min": om.min(),
                "omission_max": om.max(),
                "combined_mean": both.mean(axis=1).mean() if len(both) else np.nan,
                "score_gap_mean": (both["hallucination_score"] - both["omission_score"]).mean()
                if len(both)
                else np.nan,
                "hallucination_omission_r": corr_r,
                "hallucination_omission_p": corr_p,
            }
        )

    return pd.DataFrame(rows)


def make_summary_bar_fig(summary_df: pd.DataFrame) -> plt.Figure:
    labels = summary_df["display_name"].tolist()
    x = np.arange(len(summary_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        summary_df["hallucination_mean"],
        width,
        yerr=summary_df["hallucination_std"],
        capsize=4,
        label="Hallucination",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        summary_df["omission_mean"],
        width,
        yerr=summary_df["omission_std"],
        capsize=4,
        label="Omission",
        color="darkorange",
        alpha=0.75,
    )
    ax.set_ylabel("Mean Agent Judge Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Agent Judge Scores by Rewrite Type (mean +/- std)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def make_combined_hist_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    """One row per prompt, two columns (hallucination / omission).

    Styling matches plot_judge_scores_grid.py::make_combined_hist_fig.
    """
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
        for col_i, (score_col, label) in enumerate(SCORE_COLS):
            ax = axes[row_i][col_i]
            vals = sub[score_col].dropna()
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

    fig.tight_layout()
    return fig


def make_score_pair_scatter_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    n_cols = 2
    n_rows = math.ceil(len(rewrite_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    for idx, rcol in enumerate(rewrite_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = results_df[results_df["rewrite_col"] == rcol][["hallucination_score", "omission_score"]].dropna()
        x = sub["hallucination_score"]
        y = sub["omission_score"]

        ax.scatter(x, y, alpha=0.65, s=55, color="steelblue")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)

        if len(sub) >= 3 and x.nunique() > 1 and y.nunique() > 1:
            r, p = stats.pearsonr(x, y)
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m * xs + b, color="red", linewidth=2.0)
            title = f"{rename_prompt(rcol)}\nr={r:.2f}, p={p:.3f}"
        else:
            title = rename_prompt(rcol)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Hallucination Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Omission Score", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=11)

    for idx in range(len(rewrite_cols), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    return fig



def make_model_scatter_grid_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    """All (rewrite_col x score_pair) scatter plots in a 2-column grid.

    Styling matches plot_judge_scores_grid.py::make_combined_scatter_fig.
    """
    n_grid_cols = 2
    all_cells = [
        (rcol, sp)
        for rcol in rewrite_cols
        for sp in MODEL_SCORE_PAIRS
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

    fig.tight_layout()
    return fig


def make_correlation_model_vs_judge_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    """Old-style 2 x N correlation panel: agent judge score vs detector score."""
    n_rows = len(MODEL_SCORE_PAIRS)
    n_cols = len(rewrite_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for row_i, (x_col, y_col, x_label) in enumerate(MODEL_SCORE_PAIRS):
        for col_i, rcol in enumerate(rewrite_cols):
            ax = axes[row_i][col_i]
            sub = results_df[results_df["rewrite_col"] == rcol][[x_col, y_col]].dropna()
            ax.scatter(sub[x_col], sub[y_col], alpha=0.5, s=20)

            if len(sub) >= 3 and sub[x_col].nunique() > 1 and sub[y_col].nunique() > 1:
                r, p = stats.pearsonr(sub[x_col], sub[y_col])
                m, b = np.polyfit(sub[x_col], sub[y_col], 1)
                xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
                ax.plot(xs, m * xs + b, color="red", linewidth=1.2)
                ax.set_title(f"{rcol}\nr={r:.2f}, p={p:.3f}", fontsize=9)
            else:
                ax.set_title(f"{rcol}\nn={len(sub)}", fontsize=9)

            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel("Model Score (mean eligible folds)", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)

    fig.suptitle("Correlation: detector model score vs coding-agent judge scores", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig

def make_gap_hist_fig(results_df: pd.DataFrame, rewrite_cols: list[str]) -> plt.Figure:
    n_cols = 2
    n_rows = math.ceil(len(rewrite_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)

    for idx, rcol in enumerate(rewrite_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = results_df[results_df["rewrite_col"] == rcol][["hallucination_score", "omission_score"]].dropna()
        gap = sub["hallucination_score"] - sub["omission_score"]
        ax.hist(gap, bins=20, range=(-1, 1), color="purple", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(gap.mean(), color="black", linestyle="-", linewidth=1.3)
        ax.set_xlim(-1, 1)
        ax.set_title(f"{rename_prompt(rcol)}\nmean gap={gap.mean():.3f}", fontsize=13)
        ax.set_xlabel("Hallucination Score - Omission Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.tick_params(labelsize=10)

    for idx in range(len(rewrite_cols), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    return fig


def save_individual_hists(results_df: pd.DataFrame, rewrite_cols: list[str], hists_dir: Path) -> None:
    hists_dir.mkdir(parents=True, exist_ok=True)

    for rcol in rewrite_cols:
        sub = results_df[results_df["rewrite_col"] == rcol]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (score_col, label) in zip(axes, SCORE_COLS):
            vals = sub[score_col].dropna()
            ax.hist(vals, bins=20, range=(0, 1), color="steelblue", alpha=0.75, edgecolor="white")
            ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.2)
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1)
            ax.set_title(f"{label}\nn={len(vals)}, mean={vals.mean():.3f}")

        fig.suptitle(f"Agent Score Distributions - {rename_prompt(rcol)}")
        fig.tight_layout()
        out_path = hists_dir / f"{rcol.replace('/', '_').replace(' ', '_')}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"  Saved hist -> {out_path}")


def write_outputs(results_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rewrite_cols = ordered_rewrite_cols(results_df)
    summary_df = make_summary_table(results_df, rewrite_cols)

    summary_csv = out_dir / "agent_judge_score_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved -> {summary_csv}")

    hists_grid_pdf = out_dir / "agent_judge_hists_grid.pdf"
    fig_hists = make_combined_hist_fig(results_df, rewrite_cols)
    fig_hists.savefig(hists_grid_pdf, dpi=150, bbox_inches="tight", format="pdf")
    plt.close(fig_hists)
    print(f"Saved -> {hists_grid_pdf}")

    pair_grid_pdf = out_dir / "agent_judge_score_pair_grid.pdf"
    fig_pair = make_score_pair_scatter_fig(results_df, rewrite_cols)
    fig_pair.savefig(pair_grid_pdf, dpi=150, bbox_inches="tight", format="pdf")
    plt.close(fig_pair)
    print(f"Saved -> {pair_grid_pdf}")

    combined_pdf = out_dir / "agent_judge_plots.pdf"
    with PdfPages(combined_pdf) as pdf:
        figs = [
            make_summary_bar_fig(summary_df),
            make_combined_hist_fig(results_df, rewrite_cols),
            make_score_pair_scatter_fig(results_df, rewrite_cols),
            make_gap_hist_fig(results_df, rewrite_cols),
        ]
        if "mean_model_score" in results_df.columns and results_df["mean_model_score"].notna().any():
            model_rewrite_cols = [
                rcol
                for rcol in rewrite_cols
                if results_df.loc[results_df["rewrite_col"] == rcol, "mean_model_score"].notna().any()
            ]
            figs.extend([
                make_model_scatter_grid_fig(results_df, model_rewrite_cols),
                make_correlation_model_vs_judge_fig(results_df, model_rewrite_cols),
            ])
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved -> {combined_pdf}")

    if "mean_model_score" in results_df.columns and results_df["mean_model_score"].notna().any():
        model_rewrite_cols = [
            rcol
            for rcol in rewrite_cols
            if results_df.loc[results_df["rewrite_col"] == rcol, "mean_model_score"].notna().any()
        ]
        model_grid_pdf = out_dir / "agent_judge_model_scatter_grid.pdf"
        fig_model_grid = make_model_scatter_grid_fig(results_df, model_rewrite_cols)
        fig_model_grid.savefig(model_grid_pdf, dpi=150, bbox_inches="tight", format="pdf")
        plt.close(fig_model_grid)
        print(f"Saved -> {model_grid_pdf}")

        correlation_pdf = out_dir / "correlation_model_vs_agent_judge.pdf"
        fig_corr = make_correlation_model_vs_judge_fig(results_df, model_rewrite_cols)
        fig_corr.savefig(correlation_pdf, dpi=150, bbox_inches="tight", format="pdf")
        plt.close(fig_corr)
        print(f"Saved -> {correlation_pdf}")

    print(f"Saving per-rewrite histograms to {out_dir / 'hists'} ...")
    save_individual_hists(results_df, rewrite_cols, out_dir / "hists")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-parquet",
        type=Path,
        default=DEFAULT_RESULTS_PARQUET,
        help="Agent judge parquet to plot.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated PDFs and CSV summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.results_parquet} ...")
    results_df = load_results(args.results_parquet)
    rewrite_cols = ordered_rewrite_cols(results_df)
    print(f"Rewrite columns: {rewrite_cols}")
    print(f"Display names:   {[rename_prompt(col) for col in rewrite_cols]}")
    write_outputs(results_df, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
