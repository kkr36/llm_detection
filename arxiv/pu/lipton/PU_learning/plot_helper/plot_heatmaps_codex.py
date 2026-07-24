"""Codex heatmaps: same form as plot_helper/plot_heatmaps.py::make_heatmap, but

  * Codex is added as a new train row (PN block) and new test column,
  * the "all" column is dropped (its models predate Codex and weren't retrained),
  * the PNU row is dropped for the same reason (no Codex PNU models).

Reads logging_accuracy_llm_codex_remade.csv (produced by prepare_heatmap_codex.py)
and writes one heatmap PDF per metric, mirroring the original styling.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

# reuse styling + metric transforms from the original plotter
from plot_heatmaps import (
    orange_white_purple, fmt, binary_metrics, diverging_metrics, name_to_name,
    add_accuracy_cols, reverse_bias, reverse_plugin, plot_metrics,
)

matplotlib.rc('font', **{'weight': 'bold', 'size': 37})

ci = False  # show CI text in annotations

# Column/row order for the non-gemini codex matrix (no "all"); Codex last.
LLMS_CODEX = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen", "Codex"]

label_rename_codex = {
    "GPT OSS 120b": "GPT",
    "Llama 3.3 70b Instruct": "Llama",
    "Qwen": "Qwen",
    "Gemini 3 Preview": "Gemini 3",
    "Codex": "Codex",
}

input_file = "../logging_accuracy_llm_codex_remade.csv"
output_folder = input_file.split("/")[-1].split(".csv")[0] + "_paper"
os.makedirs(output_folder, exist_ok=True)


def build_heatmap_df(df, metric, col_order, ci_level=0.95):
    """PN block (col_order x col_order) + Avg off-diagonal row + PU diagonal row."""
    lower_col = f"{metric}_l_{ci_level}"
    upper_col = f"{metric}_u_{ci_level}"

    # -------- PN block --------
    pn = df[df["learning_method"] == "PN"]

    def pivot_metric(col):
        return (
            pn.pivot(index="train_llm", columns="test_llm", values=col)
            .reindex(index=col_order, columns=col_order)
        )

    pn_point = pivot_metric(metric)
    pn_lower = pivot_metric(lower_col)
    pn_upper = pivot_metric(upper_col)

    # -------- PU diagonal --------
    pu = df[df["learning_method"] == "PU"]
    pu_diag = pu[pu["train_llm"] == pu["test_llm"]]
    pu_point = pd.DataFrame(np.nan, index=["PU + TTA"], columns=col_order)
    pu_lower = pu_point.copy()
    pu_upper = pu_point.copy()
    for _, row in pu_diag.iterrows():
        llm = row["train_llm"]
        if llm in col_order:
            pu_point.loc["PU + TTA", llm] = row[metric]
            pu_lower.loc["PU + TTA", llm] = row[lower_col]
            pu_upper.loc["PU + TTA", llm] = row[upper_col]

    # -------- Off-diagonal PN average --------
    pn_no_diag_point = pn_point.copy()
    pn_no_diag_lower = pn_lower.copy()
    pn_no_diag_upper = pn_upper.copy()
    np.fill_diagonal(pn_no_diag_point.values, np.nan)
    np.fill_diagonal(pn_no_diag_lower.values, np.nan)
    np.fill_diagonal(pn_no_diag_upper.values, np.nan)

    avg_point = pn_no_diag_point.mean(axis=0, skipna=True)
    avg_lower = pn_no_diag_lower.mean(axis=0, skipna=True)
    avg_upper = pn_no_diag_upper.mean(axis=0, skipna=True)
    avg_point.name = avg_lower.name = avg_upper.name = "Avg. Supervised OOD"

    point_df = pd.concat([pn_point, avg_point.to_frame().T, pu_point])
    lower_df = pd.concat([pn_lower, avg_lower.to_frame().T, pu_lower])
    upper_df = pd.concat([pn_upper, avg_upper.to_frame().T, pu_upper])
    return point_df, lower_df, upper_df


def make_heatmap_codex(df, metrics, title=False):
    col_order = LLMS_CODEX
    n_llm = len(col_order)

    for metric in metrics:
        point_df, lower_df, upper_df = build_heatmap_df(df, metric, col_order, ci_level=0.95)

        plot_df = point_df.copy()
        if metric == "bbe":
            plot_df = plot_df - 0.5
            lower_df = lower_df - 0.5
            upper_df = upper_df - 0.5

        # ---- annotations ----
        annot = plot_df.copy().astype(str)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                val = plot_df.iloc[i, j]
                lo, hi = lower_df.iloc[i, j], upper_df.iloc[i, j]
                if pd.isna(val):
                    annot.iloc[i, j] = ""
                else:
                    annot.iloc[i, j] = (f"{fmt(val)}\n[{fmt(lo)}, {fmt(hi)}]" if ci else f"{fmt(val)}")

        plot_df = plot_df.rename(index=label_rename_codex, columns=label_rename_codex)
        annot = annot.rename(index=label_rename_codex, columns=label_rename_codex)

        plt.figure(figsize=(22, 16))

        if metric in binary_metrics:
            cmap = "YlOrBr"
            data_min, data_max = np.nanmin(plot_df.values), np.nanmax(plot_df.values)
            margin = max((data_max - data_min) * 0.05, 0.01)
            vmin, vmax = max(0.0, data_min - margin), min(1.0, data_max + margin)
            center = (vmin + vmax) / 2
        else:  # diverging
            cmap = orange_white_purple
            if metric == "bbe":
                center = 0.0
                max_dev = np.nanmax(np.abs(plot_df.values - center))
                vmin, vmax = center - max_dev, center + max_dev
            else:
                data_min, data_max = np.nanmin(plot_df.values), np.nanmax(plot_df.values)
                margin = max((data_max - data_min) * 0.05, 0.01)
                vmin, vmax = max(0.0, data_min - margin), min(1.0, data_max + margin)
                center = 0.5

        ax = sns.heatmap(plot_df, annot=annot, fmt="", cmap=cmap,
                         center=center, vmin=vmin, vmax=vmax)
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x))
        )

        # Layout rows: [n_llm PN rows][avg row][PU row]
        n_total = n_llm + 2
        n_cols = len(plot_df.columns)

        # Thick line between PN block and the avg row
        ax.axhline(y=n_llm, color="black", linewidth=4, zorder=4)

        # White gap + thin lines separating the avg row from the PU (TTA) row
        sep_y = n_llm + 1  # boundary between avg (row n_llm) and PU (row n_llm+1)
        gap_h = 0.20
        ax.add_patch(plt.Rectangle(
            (0, sep_y - gap_h / 2), n_cols, gap_h,
            facecolor="white", edgecolor="none",
            transform=ax.transData, clip_on=True, zorder=3,
        ))
        ax.axhline(y=sep_y - gap_h / 2, color="black", linewidth=1, zorder=4)
        ax.axhline(y=sep_y + gap_h / 2, color="black", linewidth=1, zorder=4)

        # Bracket labeling the supervised rows (PN block + avg row)
        n_supervised = n_llm + 1
        bracket_top = 1.0
        bracket_bot = 1.0 - n_supervised / n_total
        bx, tick = -0.3, 0.025
        for yfrac in [bracket_top, bracket_bot]:
            ax.plot([bx, bx + tick], [yfrac, yfrac],
                    transform=ax.transAxes, clip_on=False, color="black", lw=2.5)
        ax.plot([bx, bx], [bracket_bot, bracket_top],
                transform=ax.transAxes, clip_on=False, color="black", lw=2.5)
        ax.text(bx - 0.03, (bracket_top + bracket_bot) / 2, "Supervised\nLearning",
                transform=ax.transAxes, ha="right", va="center", rotation=90,
                fontsize=30, fontweight="bold")

        if title:
            plt.title(name_to_name.get(metric, metric))
        plt.xlabel("Test LLM")
        plt.ylabel("Train LLM / Method")

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{save_folder}/heatmap_{metric}_ci.pdf", format="pdf", bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    data = pd.read_csv(input_file)
    data = add_accuracy_cols(data)
    data = reverse_bias(reverse_plugin(data))
    make_heatmap_codex(data, plot_metrics, title=False)
    print(f"wrote heatmaps to {output_folder}/")
