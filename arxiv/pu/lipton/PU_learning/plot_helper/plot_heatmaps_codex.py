"""Codex heatmaps: same form as plot_helper/plot_heatmaps.py::make_heatmap, but

  * Codex is added as a new train row (PN block) and new test column,
  * the "all" column is dropped (its models predate Codex and weren't retrained),
  * the PNU row is dropped for the same reason (no Codex PNU models).

Reads logging_accuracy_llm_codex_remade.csv (produced by prepare_heatmap_codex.py)
and writes one heatmap PDF per metric, mirroring the original styling.
"""
import os
import math
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

# Grid panel order — identical to the reference gemini grid (plot_heatmaps.py's
# gemini plot_metrics): includes AI Recall (tnr) as a panel.
GRID_METRICS = ["auc", "accuracy", "pos_prob", "neg_prob", "bce", "tpr", "tnr", "bbe", "plugin-int"]
# Same order, but excluding tnr -> 8 panels.
GRID_METRICS_NO_TNR = [m for m in GRID_METRICS if m != "tnr"]

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

        # Bracket labeling only the supervised PN rows (Llama .. Codex), so the
        # label wraps just the train-LLM block and does not intersect the
        # "Avg. Supervised OOD" row text below it.
        n_supervised = n_llm
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


def make_heatmap_ci_codex(df, metrics, title=False, point_fontsize=30, ci_fontsize=25):
    """Codex version of plot_heatmaps.make_heatmap_ci: single heatmap per metric with
    the point estimate + 95% CI drawn in each cell (two font sizes). No "all", no PNU.
    Used for the standalone AI-recall (tnr) figure -> heatmap_{metric}_with_ci.pdf."""
    col_order = LLMS_CODEX
    n_llm = len(col_order)

    for metric in metrics:
        point_df, lower_df, upper_df = build_heatmap_df(df, metric, col_order, ci_level=0.95)

        plot_df = point_df.copy()
        if metric == "bbe":
            plot_df = plot_df - 0.5
            lower_df = lower_df - 0.5
            upper_df = upper_df - 0.5

        # keep raw (pre-rename) copies for manual per-cell annotation
        raw_point = plot_df.copy()
        raw_lower = lower_df.copy()
        raw_upper = upper_df.copy()

        plot_df = plot_df.rename(index=label_rename_codex, columns=label_rename_codex)

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

        ax = sns.heatmap(plot_df, annot=False, fmt="", cmap=cmap,
                         center=center, vmin=vmin, vmax=vmax)

        # point estimate + CI text, separate font sizes
        for i in range(raw_point.shape[0]):
            for j in range(raw_point.shape[1]):
                val = raw_point.iloc[i, j]
                if pd.isna(val):
                    continue
                lo, hi = raw_lower.iloc[i, j], raw_upper.iloc[i, j]
                cx, cy = j + 0.5, i + 0.5
                ax.text(cx, cy - 0.15, fmt(val), ha="center", va="center",
                        fontsize=point_fontsize, fontweight="bold", color="black")
                ax.text(cx, cy + 0.28, f"[{fmt(lo)},{fmt(hi)}]", ha="center", va="center",
                        fontsize=ci_fontsize, color="black")

        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x))
        )

        # Layout rows: [n_llm PN rows][avg row][PU row]
        n_total = n_llm + 2
        n_cols = len(plot_df.columns)

        # Thick line between PN block and avg row
        ax.axhline(y=n_llm, color="black", linewidth=4, zorder=4)

        # White gap + thin lines between avg row and PU (TTA) row
        sep_y = n_llm + 1
        gap_h = 0.20
        ax.add_patch(plt.Rectangle(
            (0, sep_y - gap_h / 2), n_cols, gap_h,
            facecolor="white", edgecolor="none",
            transform=ax.transData, clip_on=True, zorder=3,
        ))
        ax.axhline(y=sep_y - gap_h / 2, color="black", linewidth=1, zorder=4)
        ax.axhline(y=sep_y + gap_h / 2, color="black", linewidth=1, zorder=4)

        # Bracket labeling only the supervised PN rows (Llama .. Codex), so the
        # label wraps just the train-LLM block and does not intersect the
        # "Avg. Supervised OOD" row text below it.
        n_supervised = n_llm
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
        plt.savefig(f"{save_folder}/heatmap_{metric}_with_ci.pdf", format="pdf", bbox_inches="tight")
        plt.clf()


def make_heatmap_grid_codex(df, metrics, title=True, fname="heatmap_grid.pdf"):
    """Codex version of plot_heatmaps.make_heatmap_grid: one PDF with a subplot per
    metric. No "all", no PNU row. Saves titled/<fname> (title=True)."""
    col_order = LLMS_CODEX
    n_llm = len(col_order)

    n = len(metrics)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    with matplotlib.rc_context({'font.size': 10 + 10, 'font.weight': 'bold'}):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 9 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            point_df, lower_df, upper_df = build_heatmap_df(df, metric, col_order, ci_level=0.95)

            plot_df = point_df.copy()
            if metric in ["bbe", "plugin-int"]:
                plot_df = plot_df - 0.5
                lower_df = lower_df - 0.5
                upper_df = upper_df - 0.5

            annot = plot_df.copy().astype(str)
            for i in range(plot_df.shape[0]):
                for j in range(plot_df.shape[1]):
                    val = plot_df.iloc[i, j]
                    lo, hi = lower_df.iloc[i, j], upper_df.iloc[i, j]
                    annot.iloc[i, j] = "" if pd.isna(val) else (
                        f"{fmt(val)}\n[{fmt(lo)}, {fmt(hi)}]" if ci else f"{fmt(val)}"
                    )

            plot_df_r = plot_df.rename(index=label_rename_codex, columns=label_rename_codex)
            annot_r = annot.rename(index=label_rename_codex, columns=label_rename_codex)

            if metric in binary_metrics:
                cmap = "YlOrBr_r" if metric in ("bce", "neg_prob") else "YlOrBr"
                data_min, data_max = np.nanmin(plot_df_r.values), np.nanmax(plot_df_r.values)
                margin = max((data_max - data_min) * 0.05, 0.01)
                vmin, vmax = max(0.0, data_min - margin), min(1.0, data_max + margin)
                center = (vmin + vmax) / 2
            else:
                cmap = orange_white_purple
                if metric in ["bbe", "plugin-int"]:
                    center = 0.0
                    max_dev = np.nanmax(np.abs(plot_df_r.values - center))
                    vmin, vmax = center - max_dev, center + max_dev
                else:
                    data_min, data_max = np.nanmin(plot_df_r.values), np.nanmax(plot_df_r.values)
                    margin = max((data_max - data_min) * 0.05, 0.01)
                    vmin, vmax = max(0.0, data_min - margin), min(1.0, data_max + margin)
                    center = 0.5

            sns.heatmap(
                plot_df_r, annot=annot_r, fmt="", cmap=cmap,
                center=center, vmin=vmin, vmax=vmax,
                ax=ax, annot_kws={"size": 7 + 10},
            )
            ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x))
            )

            # Layout rows: [n_llm PN rows][avg row][PU row]
            _n_total = len(plot_df_r)
            _n_cols_h = len(plot_df_r.columns)
            _sep_y = _n_total - 1  # boundary between avg and PU row
            _gap_h = 0.20
            ax.add_patch(plt.Rectangle(
                (0, _sep_y - _gap_h / 2), _n_cols_h, _gap_h,
                facecolor="white", edgecolor="none",
                transform=ax.transData, clip_on=True, zorder=3,
            ))
            ax.axhline(y=_sep_y - _gap_h / 2, color="black", linewidth=0.8, zorder=4)
            ax.axhline(y=_sep_y + _gap_h / 2, color="black", linewidth=0.8, zorder=4)
            ax.axhline(y=_n_total - 2, color="black", linewidth=2, zorder=4)  # PN | avg

            ax.set_title(name_to_name.get(metric, metric), fontsize=12 + 15, fontweight="bold")
            ax.set_xlabel("Test LLM", fontsize=9 + 15)
            ax.set_ylabel("Train LLM / Method", fontsize=9 + 15)
            ax.tick_params(labelsize=7 + 10)

        for idx in range(len(metrics), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{save_folder}/{fname}", format="pdf", bbox_inches="tight")
        plt.clf()
        plt.close(fig)


if __name__ == "__main__":
    data = pd.read_csv(input_file)
    data = add_accuracy_cols(data)
    data = reverse_bias(reverse_plugin(data))

    # Non-grid figures, one per metric: a without-CI version (heatmap_{m}_ci.pdf)
    # and a with-CI version (heatmap_{m}_with_ci.pdf). Both use the corrected
    # bracket that wraps only the Llama..Codex train rows.
    make_heatmap_codex(data, GRID_METRICS, title=False)     # point-estimate only
    make_heatmap_ci_codex(data, GRID_METRICS, title=False)  # with per-cell CIs

    # Grid figures: full 9-metric grid, plus an 8-metric grid excluding tnr.
    make_heatmap_grid_codex(data, GRID_METRICS, title=True,
                            fname="heatmap_grid.pdf")
    make_heatmap_grid_codex(data, GRID_METRICS_NO_TNR, title=True,
                            fname="heatmap_grid_no_tnr.pdf")

    print(f"wrote heatmaps to {output_folder}/ "
          f"(heatmap_{{metric}}_ci.pdf, heatmap_{{metric}}_with_ci.pdf, "
          f"titled/heatmap_grid.pdf, titled/heatmap_grid_no_tnr.pdf)")
