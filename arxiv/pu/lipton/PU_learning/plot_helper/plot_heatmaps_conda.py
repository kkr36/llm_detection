"""Heatmaps for ConDA models (single panel).

Rows = source LLM1, columns = target LLM2, cell = eval metric of the ConDA model
trained with {LLM1, LLM2} and evaluated on LLM2. Reuses the pairwise-pivot logic
from plot_heatmaps_pnu.py (the "evaluated on LLM2" panel).
"""
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import math
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

font = {"weight": "bold", "size": 30}
matplotlib.rc("font", **font)

orange_white_purple = LinearSegmentedColormap.from_list(
    "orange_white_purple", ["orange", "white", "purple"][::-1]
)

# Base 4x4 sweep + the Codex sweep (Codex as a 5th row/column).
INPUT_FILES = ["../logging_accuracy_llm_conda.csv", "../logging_accuracy_llm_conda_codex.csv"]
OUTPUT_FOLDER = "logging_accuracy_llm_conda"

TRAIN_TO_SPACE = {
    "GPT_OSS_120b": "GPT OSS 120b",
    "Gemini_3_Preview": "Gemini 3 Preview",
    "Llama_3.3_70b_Instruct": "Llama 3.3 70b Instruct",
    "Qwen": "Qwen",
    "Codex": "Codex",
    "all": "all",
}
DISPLAY = {
    "GPT OSS 120b": "GPT",
    "Gemini 3 Preview": "Gemini 3",
    "Llama 3.3 70b Instruct": "Llama",
    "Qwen": "Qwen",
    "Codex": "Codex 5.5",
    "all": "all",
}
LLM_ORDER = ["GPT OSS 120b", "Gemini 3 Preview", "Llama 3.3 70b Instruct", "Qwen", "Codex"]

PLOT_METRICS = ["auc", "accuracy", "tpr", "tnr", "bce", "bbe", "plugin-int"]
BINARY_METRICS = {"auc", "accuracy", "pos_prob", "neg_prob", "bce", "tpr", "tnr"}
# Per-box titles kept identical to plot_heatmaps.py::name_to_name (the codex grid).
NAME_TO_NAME = {
    "auc": "AUC", "accuracy": "Bal. Accuracy", "tpr": "Human Recall",
    "tnr": "AI Recall", "pos_prob": "Avg. P(human | human)",
    "neg_prob": "Avg. P(human | AI)", "bce": "Bal. Cross-Entropy",
    "bbe": "Bias", "plugin-int": "Bias Avg P(AI)",
}

# Unseen-generalization grid: metrics averaged over the LLMs NOT seen in training.
UNSEEN_CSV = "../logging_accuracy_llm_conda_unseen.csv"
# Grid panel order — identical to the reference gemini grid (heatmap_grid.pdf).
UNSEEN_METRICS = ["auc", "accuracy", "pos_prob", "neg_prob", "bce", "tpr", "tnr", "bbe", "plugin-int"]


def fmt(v):
    if pd.isna(v) or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = f"{v:.2f}"
    if s.startswith("0."):
        return s[1:]
    elif s.startswith("-0."):
        return "-" + s[2:]
    return s


def add_accuracy_cols(df, ci_level=0.95):
    df = df.copy()
    ci = str(ci_level)
    tpr, fpr = df["tpr"], df["fpr"]
    tpr_l = df[f"tpr_l_{ci}"] if f"tpr_l_{ci}" in df.columns else tpr
    tpr_u = df[f"tpr_u_{ci}"] if f"tpr_u_{ci}" in df.columns else tpr
    fpr_l = df[f"fpr_l_{ci}"] if f"fpr_l_{ci}" in df.columns else fpr
    fpr_u = df[f"fpr_u_{ci}"] if f"fpr_u_{ci}" in df.columns else fpr
    df["accuracy"] = (tpr + 1 - fpr) / 2
    df[f"accuracy_l_{ci}"] = (tpr_l + 1 - fpr_u) / 2
    df[f"accuracy_u_{ci}"] = (tpr_u + 1 - fpr_l) / 2
    return df


def reverse_bias(df, ci_level=0.95):
    df = df.copy()
    ci = str(ci_level)
    for c in ["bbe", f"bbe_l_{ci}", f"bbe_u_{ci}"]:
        df[c] = 1 - df[c]
    return df


def reverse_plugin(df, ci_level=0.95):
    df = df.copy()
    ci = str(ci_level)
    for c in ["plugin-int", f"plugin-int_l_{ci}", f"plugin-int_u_{ci}"]:
        df[c] = 1 - df[c]
    return df


def build_pivot(df, metric, ci_level=0.95):
    """rows = llm1_norm (source), columns = llm2_norm (target); value on LLM2."""
    lower_col = f"{metric}_l_{ci_level}"
    upper_col = f"{metric}_u_{ci_level}"
    sub = df[df["test_llm"] == df["llm2_norm"]]

    def _pivot(col):
        return (
            sub.pivot_table(index="llm1_norm", columns="llm2_norm", values=col, aggfunc="first")
            .reindex(index=LLM_ORDER, columns=LLM_ORDER)
        )

    point, lower, upper = _pivot(metric), _pivot(lower_col), _pivot(upper_col)
    for llm in LLM_ORDER:
        if llm in point.index and llm in point.columns:
            point.loc[llm, llm] = np.nan
            lower.loc[llm, llm] = np.nan
            upper.loc[llm, llm] = np.nan
    return point, lower, upper


def _cmap_settings(metric, plot_df):
    vals = plot_df.values.astype(float)
    if metric in BINARY_METRICS:
        # Reverse cmap for lower-is-better metrics, matching the reference grid.
        cmap = "YlOrBr_r" if metric in ("bce", "neg_prob") else "YlOrBr"
        dmin, dmax = np.nanmin(vals), np.nanmax(vals)
        margin = max((dmax - dmin) * 0.05, 0.01)
        vmin, vmax = max(0.0, dmin - margin), min(1.0, dmax + margin)
        center = (vmin + vmax) / 2
    elif metric == "bbe":
        cmap = orange_white_purple
        center = 0.0
        max_dev = np.nanmax(np.abs(vals - center))
        vmin, vmax = center - max_dev, center + max_dev
    else:
        cmap = orange_white_purple
        dmin, dmax = np.nanmin(vals), np.nanmax(vals)
        margin = max((dmax - dmin) * 0.05, 0.01)
        vmin, vmax = max(0.0, dmin - margin), min(1.0, dmax + margin)
        center = 0.5
    return cmap, vmin, vmax, center


def make_heatmaps(df, metrics, title=True):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for metric in metrics:
        point_df, lower_df, upper_df = build_pivot(df, metric)

        plot_df = point_df.copy()
        if metric == "bbe":
            plot_df = plot_df - 0.5

        annot = plot_df.copy().astype(object)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                val = plot_df.iloc[i, j]
                annot.iloc[i, j] = "" if pd.isna(val) else fmt(val)

        plot_df_r = plot_df.rename(index=DISPLAY, columns=DISPLAY)
        annot_r = annot.rename(index=DISPLAY, columns=DISPLAY)
        cmap, vmin, vmax, center = _cmap_settings(metric, plot_df_r)

        fig, ax = plt.subplots(figsize=(17, 14))
        sns.heatmap(plot_df_r, annot=annot_r, fmt="", cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, ax=ax, annot_kws={"size": 26})
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x))
        )
        ax.set_xlabel("Target LLM 2 (eval)", fontsize=26, fontweight="bold")
        ax.set_ylabel("Source LLM 1", fontsize=26, fontweight="bold")
        ax.tick_params(labelsize=22)
        if title:
            ax.set_title(f"{NAME_TO_NAME.get(metric, metric)} — ConDA (eval on LLM2)",
                         fontsize=27, fontweight="bold")

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_FOLDER, f"heatmap_conda_{metric}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.clf()
        plt.close(fig)
        print(f"Saved {save_path}")


def make_seen_heatmap_grid(df, metrics=PLOT_METRICS, ci=False, title=True):
    """Grid of NxN heatmaps (one panel per metric), ConDA evaluated on the target
    LLM2 (rows = source LLM1, cols = target LLM2). Same content as make_heatmaps but
    collected into a single figure. Saved as heatmap_eval_on_llm2_grid.pdf."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    n = len(metrics)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 9 * nrows), squeeze=False)
    axf = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axf[idx]
        point, lower, upper = build_pivot(df, metric)
        plot_df = point.copy()
        if metric == "bbe":
            plot_df = plot_df - 0.5
            lower = lower - 0.5
            upper = upper - 0.5

        annot = plot_df.copy().astype(object)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                v = plot_df.iloc[i, j]
                if pd.isna(v):
                    annot.iloc[i, j] = ""
                elif ci and not pd.isna(lower.iloc[i, j]):
                    annot.iloc[i, j] = f"{fmt(v)}\n[{fmt(lower.iloc[i, j])}, {fmt(upper.iloc[i, j])}]"
                else:
                    annot.iloc[i, j] = fmt(v)

        plot_df_r = plot_df.rename(index=DISPLAY, columns=DISPLAY)
        annot_r = annot.rename(index=DISPLAY, columns=DISPLAY)
        cmap, vmin, vmax, center = _cmap_settings(metric, plot_df_r)

        sns.heatmap(plot_df_r, annot=annot_r, fmt="", cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, ax=ax, annot_kws={"size": 15},
                    cbar_kws={"shrink": 0.8})
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x)))
        ax.set_title(NAME_TO_NAME.get(metric, metric), fontsize=22, fontweight="bold")
        ax.set_xlabel("Target LLM 2 (eval)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Source LLM 1", fontsize=16, fontweight="bold")
        ax.tick_params(labelsize=13)

    for k in range(n, len(axf)):
        axf[k].set_visible(False)

    if title:
        fig.suptitle("ConDA — evaluated on the target LLM2 (source LLM1 x target LLM2)",
                     fontsize=24, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98] if title else None)
    save_path = os.path.join(OUTPUT_FOLDER, "heatmap_eval_on_llm2_grid.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    print(f"Saved {save_path}")


def _prep_unseen_df(path=UNSEEN_CSV):
    """Load the unseen-eval CSV and normalize (same metric conventions as the base fig)."""
    df = pd.read_csv(path)
    df = add_accuracy_cols(df)
    df = reverse_bias(reverse_plugin(df))
    df["llm1_norm"] = df["train_llm"].str.split("|").str[0].map(TRAIN_TO_SPACE)
    df["llm2_norm"] = df["train_llm"].str.split("|").str[1].map(TRAIN_TO_SPACE)
    return df


def _unseen_pivot(df, metric, ci_level=0.95):
    """5x5 pivot (rows=source LLM1, cols=unlabeled LLM2); each cell = the metric
    AVERAGED over the models' UNSEEN test LLMs (those != LLM1 and != LLM2).
    Also returns averaged lower/upper CI bounds. Diagonal is left blank."""
    lo_col, hi_col = f"{metric}_l_{ci_level}", f"{metric}_u_{ci_level}"
    agg = {"point": (metric, "mean")}
    if lo_col in df.columns:
        agg["lo"] = (lo_col, "mean")
    if hi_col in df.columns:
        agg["hi"] = (hi_col, "mean")
    g = df.groupby(["llm1_norm", "llm2_norm"]).agg(**agg).reset_index()

    def piv(col):
        if col not in g.columns:
            return None
        return g.pivot(index="llm1_norm", columns="llm2_norm", values=col).reindex(
            index=LLM_ORDER, columns=LLM_ORDER)

    return piv("point"), piv("lo"), piv("hi")


def make_unseen_heatmap_grid(metrics=UNSEEN_METRICS, ci=True, title=True):
    """Big grid of 5x5 heatmaps (one per metric). Each cell = mean metric over the
    test LLMs unseen during that model's training. Saved as heatmap_unseen_grid.pdf."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df = _prep_unseen_df()

    n = len(metrics)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 9 * nrows), squeeze=False)
    axf = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axf[idx]
        point, lower, upper = _unseen_pivot(df, metric)
        plot_df = point.copy()
        if metric == "bbe":
            plot_df = plot_df - 0.5
            if lower is not None: lower = lower - 0.5
            if upper is not None: upper = upper - 0.5

        annot = plot_df.copy().astype(object)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                v = plot_df.iloc[i, j]
                if pd.isna(v):
                    annot.iloc[i, j] = ""
                elif ci and lower is not None and upper is not None and not pd.isna(lower.iloc[i, j]):
                    annot.iloc[i, j] = f"{fmt(v)}\n[{fmt(lower.iloc[i, j])}, {fmt(upper.iloc[i, j])}]"
                else:
                    annot.iloc[i, j] = fmt(v)

        plot_df_r = plot_df.rename(index=DISPLAY, columns=DISPLAY)
        annot_r = annot.rename(index=DISPLAY, columns=DISPLAY)
        cmap, vmin, vmax, center = _cmap_settings(metric, plot_df_r)

        sns.heatmap(plot_df_r, annot=annot_r, fmt="", cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, ax=ax, annot_kws={"size": 15},
                    cbar_kws={"shrink": 0.8})
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x)))
        ax.set_title(NAME_TO_NAME.get(metric, metric), fontsize=22, fontweight="bold")
        ax.set_xlabel("Unlabeled target LLM 2", fontsize=16, fontweight="bold")
        ax.set_ylabel("Labeled source LLM 1", fontsize=16, fontweight="bold")
        ax.tick_params(labelsize=13)

    for k in range(n, len(axf)):
        axf[k].set_visible(False)

    if title:
        fig.suptitle("ConDA generalization to UNSEEN LLMs — metric averaged over LLMs "
                     "absent from both source & target", fontsize=24, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98] if title else None)
    save_path = os.path.join(OUTPUT_FOLDER, "heatmap_unseen_grid.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    print(f"Saved {save_path}")


def _prep_seen_df():
    """Load the base + Codex sweeps (the 'eval on LLM2' rows) and normalize with the
    same metric conventions used everywhere else. One row per trained ConDA model."""
    present = [f for f in INPUT_FILES if os.path.exists(f)]
    df = pd.concat([pd.read_csv(f) for f in present], ignore_index=True)
    df = add_accuracy_cols(df)
    df = reverse_bias(reverse_plugin(df))
    df["llm1_norm"] = df["train_llm"].str.split("|").str[0].map(TRAIN_TO_SPACE)
    df["llm2_norm"] = df["train_llm"].str.split("|").str[1].map(TRAIN_TO_SPACE)
    return df


def make_seen_vs_unseen_grid(metrics=UNSEEN_METRICS, color_by="unseen", title=True):
    """Big grid of 5x5 heatmaps (one panel per metric) that contrasts, for every
    trained ConDA model {LLM1 (row), LLM2 (col)}, its performance on:
        S = the seen target LLM2 (eval on LLM2), and
        U = the mean over the UNSEEN LLMs (all test LLMs != LLM1 and != LLM2).
    Each off-diagonal cell is annotated with both numbers ("S ..\nU ..") plus their
    gap, so the seen->unseen degradation is legible per cell.

    color_by controls what the cell color encodes:
        "unseen" (default) -> the U value, using each metric's usual cmap, so the
            panel is directly comparable to heatmap_conda_<metric>.pdf (the seen grid)
            and to heatmap_unseen_grid.pdf.
        "seen"   -> the S value (same cmap conventions).
        "gap"    -> the raw (S - U) difference on a diverging cmap centered at 0.

    Saved as heatmap_seen_vs_unseen_grid.pdf in OUTPUT_FOLDER.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    seen_df = _prep_seen_df()
    unseen_df = _prep_unseen_df()

    n = len(metrics)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 9 * nrows), squeeze=False)
    axf = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axf[idx]
        seen = build_pivot(seen_df, metric)[0]           # 5x5, eval on LLM2
        unseen = _unseen_pivot(unseen_df, metric)[0]     # 5x5, mean over unseen LLMs
        if metric == "bbe":
            seen = seen - 0.5
            unseen = unseen - 0.5
        gap = seen - unseen

        if color_by == "seen":
            color_df = seen
        elif color_by == "gap":
            color_df = gap
        else:
            color_df = unseen

        annot = seen.copy().astype(object)
        for i in range(seen.shape[0]):
            for j in range(seen.shape[1]):
                s, u = seen.iloc[i, j], unseen.iloc[i, j]
                if pd.isna(s) and pd.isna(u):
                    annot.iloc[i, j] = ""
                else:
                    g = s - u
                    annot.iloc[i, j] = f"S {fmt(s)}\nU {fmt(u)}\nΔ {fmt(g)}"

        color_r = color_df.rename(index=DISPLAY, columns=DISPLAY)
        annot_r = annot.rename(index=DISPLAY, columns=DISPLAY)

        if color_by == "gap":
            cmap = orange_white_purple
            max_dev = np.nanmax(np.abs(color_r.values.astype(float)))
            max_dev = max_dev if np.isfinite(max_dev) and max_dev > 0 else 1.0
            vmin, vmax, center = -max_dev, max_dev, 0.0
        else:
            cmap, vmin, vmax, center = _cmap_settings(metric, color_r)

        sns.heatmap(color_r, annot=annot_r, fmt="", cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, ax=ax, annot_kws={"size": 13},
                    cbar_kws={"shrink": 0.8})
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x)))
        ax.set_title(NAME_TO_NAME.get(metric, metric), fontsize=22, fontweight="bold")
        ax.set_xlabel("Target LLM 2", fontsize=16, fontweight="bold")
        ax.set_ylabel("Source LLM 1", fontsize=16, fontweight="bold")
        ax.tick_params(labelsize=13)

    for k in range(n, len(axf)):
        axf[k].set_visible(False)

    if title:
        color_desc = {"seen": "seen LLM2", "gap": "seen-minus-unseen gap"}.get(
            color_by, "unseen-LLM average")
        fig.suptitle("ConDA: seen target (S) vs. mean over UNSEEN LLMs (U)  "
                     f"[color = {color_desc}]", fontsize=24, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98] if title else None)
    save_path = os.path.join(OUTPUT_FOLDER, f"heatmap_seen_vs_unseen_grid_{color_by}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    df = _prep_seen_df()

    # Grid 1: eval on the target LLM2 (same metric panels as the gap grid below).
    make_seen_heatmap_grid(df, UNSEEN_METRICS, title=True)

    # Grid 2: seen-target (S) vs. mean over UNSEEN LLMs (U), colored by the S-U gap.
    make_seen_vs_unseen_grid(UNSEEN_METRICS, color_by="gap", title=True)
