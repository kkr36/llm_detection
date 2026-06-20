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

INPUT_FILES = [
    "../logging_accuracy_llm_pnu_1.csv",
    "../logging_accuracy_llm_pnu_2.csv",
    "../logging_accuracy_llm_pnu_3.csv",
]
OUTPUT_FOLDER = "logging_accuracy_llm_pnu_combined"

TRAIN_TO_SPACE = {
    "GPT_OSS_120b": "GPT OSS 120b",
    "Gemini_3_Preview": "Gemini 3 Preview",
    "Llama_3.3_70b_Instruct": "Llama 3.3 70b Instruct",
    "Qwen": "Qwen",
    "all": "all",
}

DISPLAY = {
    "GPT OSS 120b": "GPT",
    "Gemini 3 Preview": "Gemini 3",
    "Llama 3.3 70b Instruct": "Llama",
    "Qwen": "Qwen",
    "all": "all",
}

LLM_ORDER = ["GPT OSS 120b", "Gemini 3 Preview", "Llama 3.3 70b Instruct", "Qwen"]
LLM_ORDER_WITH_ALL = LLM_ORDER + ["all"]

PLOT_METRICS = ["auc", "accuracy", "tpr", "tnr", "bce", "bbe", "plugin-int"]

BINARY_METRICS = {"auc", "accuracy", "pos_prob", "neg_prob", "bce", "tpr", "tnr"}

NAME_TO_NAME = {
    "auc": "AUC",
    "accuracy": "Bal. Accuracy",
    "tpr": "TPR",
    "tnr": "TNR",
    "bce": "Bal. Cross-Entropy",
    "bbe": "Bias",
    "plugin-int": "Bias Avg P(AI)",
}


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
    tpr = df["tpr"]
    fpr = df["fpr"]
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
    df["bbe"] = 1 - df["bbe"]
    df[f"bbe_l_{ci}"] = 1 - df[f"bbe_l_{ci}"]
    df[f"bbe_u_{ci}"] = 1 - df[f"bbe_u_{ci}"]
    return df


def reverse_plugin(df, ci_level=0.95):
    df = df.copy()
    ci = str(ci_level)
    df["plugin-int"] = 1 - df["plugin-int"]
    df[f"plugin-int_l_{ci}"] = 1 - df[f"plugin-int_l_{ci}"]
    df[f"plugin-int_u_{ci}"] = 1 - df[f"plugin-int_u_{ci}"]
    return df


def build_pivot(df, eval_on, metric, ci_level=0.95):
    """
    Build pivot: rows = llm1_norm (y-axis), columns = llm2_norm (x-axis).
    eval_on: "llm1_norm" → filter rows where test_llm == llm1_norm (model evaluated on its source LLM)
             "llm2_norm" → filter rows where test_llm == llm2_norm (model evaluated on its partner LLM)
    Both panels share identical axes so the same cell (y=x, x=y) refers to the same trained model.
    """
    lower_col = f"{metric}_l_{ci_level}"
    upper_col = f"{metric}_u_{ci_level}"

    sub = df[df["test_llm"] == df[eval_on]]

    def _pivot(col):
        return (
            sub.pivot_table(index="llm1_norm", columns="llm2_norm", values=col, aggfunc="first")
            .reindex(index=LLM_ORDER_WITH_ALL, columns=LLM_ORDER_WITH_ALL)
        )

    point = _pivot(metric)
    lower = _pivot(lower_col)
    upper = _pivot(upper_col)

    for llm in LLM_ORDER_WITH_ALL:
        if llm in point.index and llm in point.columns:
            point.loc[llm, llm] = np.nan
            lower.loc[llm, llm] = np.nan
            upper.loc[llm, llm] = np.nan

    return point, lower, upper


def _cmap_settings(metric, plot_df):
    vals = plot_df.values.astype(float)
    if metric in BINARY_METRICS:
        cmap = "YlOrBr"
        data_min = np.nanmin(vals)
        data_max = np.nanmax(vals)
        margin = max((data_max - data_min) * 0.05, 0.01)
        vmin = max(0.0, data_min - margin)
        vmax = min(1.0, data_max + margin)
        center = (vmin + vmax) / 2
    elif metric == "bbe":
        cmap = orange_white_purple
        center = 0.0
        max_dev = np.nanmax(np.abs(vals - center))
        vmin, vmax = center - max_dev, center + max_dev
    else:
        cmap = orange_white_purple
        data_min = np.nanmin(vals)
        data_max = np.nanmax(vals)
        margin = max((data_max - data_min) * 0.05, 0.01)
        vmin = max(0.0, data_min - margin)
        vmax = min(1.0, data_max + margin)
        center = 0.5
    return cmap, vmin, vmax, center


def make_heatmaps(df, metrics, title=True):
    """
    For each metric, produce a side-by-side figure:
      left panel  = heatmap 1 (x = llm1 from train pair, y = test_llm)
      right panel = heatmap 2 (x = llm2 from train pair, y = test_llm)
    Diagonal cells (where x-axis LLM == test_llm) are blank.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(34, 14))

        for ax, eval_on, panel_label in zip(
            axes,
            ["llm1_norm", "llm2_norm"],
            ["LLM1", "LLM2"],
        ):
            point_df, lower_df, upper_df = build_pivot(df, eval_on, metric)

            plot_df = point_df.copy()
            if metric == "bbe":
                plot_df = plot_df - 0.5
                lower_df = lower_df - 0.5
                upper_df = upper_df - 0.5

            annot = plot_df.copy().astype(object)
            for i in range(plot_df.shape[0]):
                for j in range(plot_df.shape[1]):
                    val = plot_df.iloc[i, j]
                    annot.iloc[i, j] = "" if pd.isna(val) else fmt(val)

            plot_df_r = plot_df.rename(index=DISPLAY, columns=DISPLAY)
            annot_r = annot.rename(index=DISPLAY, columns=DISPLAY)

            cmap, vmin, vmax, center = _cmap_settings(metric, plot_df_r)

            sns.heatmap(
                plot_df_r, annot=annot_r, fmt="",
                cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                ax=ax, annot_kws={"size": 26},
            )
            ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x))
            )
            ax.set_xlabel("Train LLM 2", fontsize=26, fontweight="bold")
            ax.set_ylabel("Train LLM 1", fontsize=26, fontweight="bold")
            ax.tick_params(labelsize=22)
            if title:
                ax.set_title(
                    f"{NAME_TO_NAME.get(metric, metric)} — evaluated on {panel_label}",
                    fontsize=27, fontweight="bold",
                )

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_FOLDER, f"heatmap_pnu_{metric}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.clf()
        plt.close(fig)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(f) for f in INPUT_FILES], ignore_index=True)
    df = add_accuracy_cols(df)
    df = reverse_bias(reverse_plugin(df))

    df["llm1"] = df["train_llm"].str.split("|").str[0]
    df["llm2"] = df["train_llm"].str.split("|").str[1]
    df["llm1_norm"] = df["llm1"].map(TRAIN_TO_SPACE)
    df["llm2_norm"] = df["llm2"].map(TRAIN_TO_SPACE)

    make_heatmaps(df, PLOT_METRICS, title=True)
