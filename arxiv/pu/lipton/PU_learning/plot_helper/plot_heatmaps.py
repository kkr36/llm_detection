from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 37
    }
import matplotlib
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

orange_white_purple = LinearSegmentedColormap.from_list(
    "orange_white_purple", ["orange", "white", "purple"][::-1]
)

ci = False  # show confidence interval text in heatmap annotations

def fmt(v):
    s = f"{v:.2f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s

label_rename_default = {
    "GPT OSS 120b": "GPT",
    "Llama 3.3 70b Instruct": "Llama",
    "Qwen": "Qwen",
    "Gemini 3 Preview": "Gemini 3"
}

_gemini_llms = ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"]
label_rename_gemini = {name: name.replace("Gemini ", "") for name in _gemini_llms}

input_file = "../logging_accuracy_llm.csv"
import os
import math
output_folder = input_file.split("/")[-1].split(".csv")[0] + "_qwen_120b_paper"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plot_metrics = [
    "auc",
    "accuracy",
    "pos_prob",
    "neg_prob",
    "bce",
    "tpr",
    "bbe",
    "plugin-int",
    # "tnr"
]
if "gemini" in input_file: plot_metrics = plot_metrics[:-2] + ["tnr"] + plot_metrics[-2:]

name_to_name = {
    "auc"        : "AUC",
    "accuracy"   : "Bal. Accuracy",
    "pos_prob"   : "Avg. P(human | human)",
    "neg_prob"   : "Avg. P(human | AI)",
    # "entropy_pos": "Shannon Entropy Human",
    # "entropy_neg": "Shannon Entropy LLM",
    # "entropy"    : "Avg. Shannon Entropy",
    "bce"        : "Bal. Cross-Entropy",
    "bbe"        : "Bias",
    "plugin-int" : "Bias Avg P(AI)",
    "tpr"        : "Human Recall",
    "tnr"        : "AI Recall"
}

binary_metrics = ["auc", "accuracy", "pos_prob", "neg_prob", "entropy_pos", "entropy_neg", "entropy", "bce"]
diverging_metrics = ["bbe", "plugin", "plugin-int"]

def add_accuracy_cols(df, ci_level=0.95):
    """Balanced accuracy = (TPR + TNR) / 2. In this file's CSV convention,
    pos_prob = TPR (Avg Pred Human) and neg_prob = FPR (Avg Pred LLM)."""
    df = df.copy()
    ci = str(ci_level)
    tpr     = df["tpr"]
    fpr     = df["fpr"]
    tpr_l   = df[f"tpr_l_{ci}"] if f"tpr_l_{ci}" in df.columns else tpr
    tpr_u   = df[f"tpr_u_{ci}"] if f"tpr_u_{ci}" in df.columns else tpr
    fpr_l   = df[f"fpr_l_{ci}"] if f"fpr_l_{ci}" in df.columns else fpr
    fpr_u   = df[f"fpr_u_{ci}"] if f"fpr_u_{ci}" in df.columns else fpr
    df["accuracy"]              = (tpr + 1 - fpr) / 2
    df[f"accuracy_l_{ci}"]     = (tpr_l + 1 - fpr_u) / 2
    df[f"accuracy_u_{ci}"]     = (tpr_u + 1 - fpr_l) / 2
    return df

def reverse_bias(df, ci_level=0.95):
    """Balanced accuracy = (TPR + TNR) / 2. In this file's CSV convention,
    pos_prob = TPR (Avg Pred Human) and neg_prob = FPR (Avg Pred LLM)."""
    df = df.copy()
    ci = str(ci_level)
    df["bbe"]              = 1-df["bbe"]
    df[f"bbe_l_{ci}"]     = 1-df[f"bbe_l_{ci}"]
    df[f"bbe_u_{ci}"]     = 1-df[f"bbe_u_{ci}"]
    return df

def reverse_plugin(df, ci_level=0.95):
    """Balanced accuracy = (TPR + TNR) / 2. In this file's CSV convention,
    pos_prob = TPR (Avg Pred Human) and neg_prob = FPR (Avg Pred LLM)."""
    df = df.copy()
    ci = str(ci_level)
    df["plugin-int"]              = 1-df["plugin-int"]
    df[f"plugin-int_l_{ci}"]     = 1-df[f"plugin-int_l_{ci}"]
    df[f"plugin-int_u_{ci}"]     = 1-df[f"plugin-int_u_{ci}"]
    return df

def make_heatmap(df, metrics, gemini, title=False):
    llms_list = ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"] if gemini else ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen"]
    llms_list += ["all"]

    def build_heatmap_df(df, metric, col_order, ci_level=0.95):

        lower_col = f"{metric}_l_{ci_level}"
        upper_col = f"{metric}_u_{ci_level}"

        # "all" gets a dedicated far-right column (PU row only), not a regular row
        col_order_no_all = [c for c in col_order if c != "all"]
        has_all = "all" in col_order

        # -------------------------
        # PN block (no "all" row/column)
        # -------------------------
        pn = df[df["learning_method"] == "PN"]

        def pivot_metric(col):
            return (
                pn.pivot(index="train_llm", columns="test_llm", values=col)
                .reindex(index=col_order_no_all, columns=col_order_no_all)
            )

        pn_point = pivot_metric(metric)
        pn_lower = pivot_metric(lower_col)
        pn_upper = pivot_metric(upper_col)

        # -------------------------
        # PU diagonal
        # -------------------------
        pu = df[df["learning_method"] == "PU"]
        pu_diag = pu[pu["train_llm"] == pu["test_llm"]]

        pu_point = pd.DataFrame(np.nan, index=["PU + TTA"], columns=col_order_no_all)
        pu_lower = pu_point.copy()
        pu_upper = pu_point.copy()

        pu_all_point = pu_all_lower = pu_all_upper = np.nan

        for _, row in pu_diag.iterrows():
            llm = row["train_llm"]
            if llm == "all" and has_all:
                pu_all_point = row[metric]
                pu_all_lower = row[lower_col]
                pu_all_upper = row[upper_col]
            elif llm in col_order_no_all:
                pu_point.loc["PU + TTA", llm] = row[metric]
                pu_lower.loc["PU + TTA", llm] = row[lower_col]
                pu_upper.loc["PU + TTA", llm] = row[upper_col]

        # -------------------------
        # Off-diagonal PN average
        # -------------------------
        pn_no_diag_point = pn_point.copy()
        pn_no_diag_lower = pn_lower.copy()
        pn_no_diag_upper = pn_upper.copy()

        np.fill_diagonal(pn_no_diag_point.values, np.nan)
        np.fill_diagonal(pn_no_diag_lower.values, np.nan)
        np.fill_diagonal(pn_no_diag_upper.values, np.nan)

        avg_point = pn_no_diag_point.mean(axis=0, skipna=True)
        avg_lower = pn_no_diag_lower.mean(axis=0, skipna=True)
        avg_upper = pn_no_diag_upper.mean(axis=0, skipna=True)

        col_order_sorted = col_order_no_all

        pn_point = pn_point.loc[col_order_sorted, col_order_sorted]
        pn_lower = pn_lower.loc[col_order_sorted, col_order_sorted]
        pn_upper = pn_upper.loc[col_order_sorted, col_order_sorted]

        pu_point = pu_point[col_order_sorted]
        pu_lower = pu_lower[col_order_sorted]
        pu_upper = pu_upper[col_order_sorted]

        avg_point = avg_point[col_order_sorted]
        avg_lower = avg_lower[col_order_sorted]
        avg_upper = avg_upper[col_order_sorted]

        avg_point.name = "Avg. Supervised OOD"
        avg_lower.name = "Avg. Supervised OOD"
        avg_upper.name = "Avg. Supervised OOD"

        # -------------------------
        # Append "all" as far-right column: PN values, avg, and PU result
        # -------------------------
        if has_all:
            pn_test_all = (
                pn[pn["test_llm"] == "all"]
                .set_index("train_llm")
                .reindex(col_order_sorted)
            )
            pn_point["all"] = pn_test_all[metric].values
            pn_lower["all"] = pn_test_all[lower_col].values
            pn_upper["all"] = pn_test_all[upper_col].values
            avg_point["all"] = pn_test_all[metric].mean(skipna=True)
            avg_lower["all"] = pn_test_all[lower_col].mean(skipna=True)
            avg_upper["all"] = pn_test_all[upper_col].mean(skipna=True)
            pu_point["all"] = pu_all_point
            pu_lower["all"] = pu_all_lower
            pu_upper["all"] = pu_all_upper

        # -------------------------
        # Combine — order: PN block, Avg (off-diag), PU
        # -------------------------
        point_df = pd.concat([pn_point, avg_point.to_frame().T, pu_point])
        lower_df = pd.concat([pn_lower, avg_lower.to_frame().T, pu_lower])
        upper_df = pd.concat([pn_upper, avg_upper.to_frame().T, pu_upper])

        return point_df, lower_df, upper_df


    # -----------------------
    # PLOT ONE HEATMAP PER METRIC
    # -----------------------
    for metric in metrics:

        point_df, lower_df, upper_df = build_heatmap_df(
            df, metric, llms_list, ci_level=0.95
        )

        # ---- Apply BBE shift to the DATA (not just annotation)
        plot_df = point_df.copy()

        if metric == "bbe":
            plot_df = plot_df - 0.5
            lower_df = lower_df - 0.5
            upper_df = upper_df - 0.5

        # ---- Build annotation matrix
        annot = plot_df.copy().astype(str)

        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):

                val = plot_df.iloc[i, j]
                lo = lower_df.iloc[i, j]
                hi = upper_df.iloc[i, j]

                if pd.isna(val):
                    annot.iloc[i, j] = ""
                else:
                    annot.iloc[i, j] = (
                        f"{fmt(val)}\n[{fmt(lo)}, {fmt(hi)}]" if ci else f"{fmt(val)}"
                    )

        label_rename = label_rename_gemini if gemini else label_rename_default
        plot_df = plot_df.rename(index=label_rename, columns=label_rename)
        annot = annot.rename(index=label_rename, columns=label_rename)

        plt.figure(figsize=(22,16))

        if metric in binary_metrics or (metric == "tnr" and gemini):
            cmap = "YlOrBr"
            data_min = np.nanmin(plot_df.values)
            data_max = np.nanmax(plot_df.values)
            margin = max((data_max - data_min) * 0.05, 0.01)
            vmin = max(0.0, data_min - margin)
            vmax = min(1.0, data_max + margin)
            center = (vmin + vmax) / 2
        else:  # diverging_metrics
            cmap = orange_white_purple
            if metric == "bbe":
                center = 0.0
                max_dev = np.nanmax(np.abs(plot_df.values - center))
                vmin, vmax = center - max_dev, center + max_dev
            else:
                data_min = np.nanmin(plot_df.values)
                data_max = np.nanmax(plot_df.values)
                margin = max((data_max - data_min) * 0.05, 0.01)
                vmin = max(0.0, data_min - margin)
                vmax = min(1.0, data_max + margin)
                center = 0.5

        ax = sns.heatmap(
            plot_df,   # <-- use shifted data
            annot=annot,
            fmt="",
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            # cbar_kws={"label": name_to_name.get(metric, metric)},
        )
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x))
        )

        # Small white gap between supervised section (PN + avg) and PU row
        _n_total = len(plot_df)
        _n_cols = len(plot_df.columns)
        _sep_y = _n_total - 1  # data-coord boundary between avg and PU rows
        _gap_h = 0.20          # gap height in data units; adjust to taste
        ax.add_patch(plt.Rectangle(
            (0, _sep_y - _gap_h / 2), _n_cols, _gap_h,
            facecolor="white", edgecolor="none",
            transform=ax.transData, clip_on=True, zorder=3,
        ))
        ax.axhline(y=_sep_y - _gap_h / 2, color="black", linewidth=1, zorder=4)
        ax.axhline(y=_sep_y + _gap_h / 2, color="black", linewidth=1, zorder=4)

        # Thick line between last LLM row and avg row
        ax.axhline(y=_n_total - 2, color="black", linewidth=4, zorder=4)

        # Thick vertical separator before the far-right "all" PU column
        if "all" in llms_list:
            ax.axvline(x=len(plot_df.columns) - 1, color="black", linewidth=4, zorder=4)

        # Bracket on y-axis labeling supervised learning rows (PN block + avg row)
        # Layout: [n_pn rows] [avg row] [sep row] [PU row]
        n_total = len(plot_df)
        n_supervised = n_total - 2  # exclude separator and PU rows
        bracket_top = 1.0
        bracket_bot = 1.0 - n_supervised / n_total
        bx = -0.3 if not gemini else -0.38  # axes fraction, left of y-axis
        tick = 0.025
        for yfrac in [bracket_top, bracket_bot]:
            ax.plot([bx, bx + tick], [yfrac, yfrac],
                    transform=ax.transAxes, clip_on=False, color="black", lw=2.5)
        ax.plot([bx, bx], [bracket_bot, bracket_top],
                transform=ax.transAxes, clip_on=False, color="black", lw=2.5)
        ax.text(bx - 0.03, (bracket_top + bracket_bot) / 2,
                "Supervised\nLearning",
                transform=ax.transAxes, ha="right", va="center", rotation=90,
                fontsize=30, fontweight="bold")

        if title:
            plt.title(name_to_name.get(metric, metric))
        plt.xlabel("Test LLM")
        plt.ylabel("Train LLM / Method")

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/heatmap_{metric}_ci.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()


def make_heatmap_ci(df, metrics, gemini, title=False, point_fontsize=30, ci_fontsize=25):
    """Like make_heatmap but always annotates each cell with point estimate + 95% CI."""
    llms_list = ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"] if gemini else ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen"]
    llms_list += ["all"]

    def build_heatmap_df(df, metric, col_order, ci_level=0.95):
        lower_col = f"{metric}_l_{ci_level}"
        upper_col = f"{metric}_u_{ci_level}"
        col_order_no_all = [c for c in col_order if c != "all"]
        has_all = "all" in col_order

        pn = df[df["learning_method"] == "PN"]

        def pivot_metric(col):
            return (
                pn.pivot(index="train_llm", columns="test_llm", values=col)
                .reindex(index=col_order_no_all, columns=col_order_no_all)
            )

        pn_point = pivot_metric(metric)
        pn_lower = pivot_metric(lower_col)
        pn_upper = pivot_metric(upper_col)

        pu = df[df["learning_method"] == "PU"]
        pu_diag = pu[pu["train_llm"] == pu["test_llm"]]
        pu_point = pd.DataFrame(np.nan, index=["PU + TTA"], columns=col_order_no_all)
        pu_lower = pu_point.copy()
        pu_upper = pu_point.copy()
        pu_all_point = pu_all_lower = pu_all_upper = np.nan

        for _, row in pu_diag.iterrows():
            llm = row["train_llm"]
            if llm == "all" and has_all:
                pu_all_point = row[metric]
                pu_all_lower = row[lower_col]
                pu_all_upper = row[upper_col]
            elif llm in col_order_no_all:
                pu_point.loc["PU + TTA", llm] = row[metric]
                pu_lower.loc["PU + TTA", llm] = row[lower_col]
                pu_upper.loc["PU + TTA", llm] = row[upper_col]

        pn_no_diag_point = pn_point.copy()
        pn_no_diag_lower = pn_lower.copy()
        pn_no_diag_upper = pn_upper.copy()
        np.fill_diagonal(pn_no_diag_point.values, np.nan)
        np.fill_diagonal(pn_no_diag_lower.values, np.nan)
        np.fill_diagonal(pn_no_diag_upper.values, np.nan)

        avg_point = pn_no_diag_point.mean(axis=0, skipna=True)
        avg_lower = pn_no_diag_lower.mean(axis=0, skipna=True)
        avg_upper = pn_no_diag_upper.mean(axis=0, skipna=True)
        col_order_sorted = col_order_no_all

        pn_point = pn_point.loc[col_order_sorted, col_order_sorted]
        pn_lower = pn_lower.loc[col_order_sorted, col_order_sorted]
        pn_upper = pn_upper.loc[col_order_sorted, col_order_sorted]
        pu_point = pu_point[col_order_sorted]
        pu_lower = pu_lower[col_order_sorted]
        pu_upper = pu_upper[col_order_sorted]
        avg_point = avg_point[col_order_sorted]
        avg_lower = avg_lower[col_order_sorted]
        avg_upper = avg_upper[col_order_sorted]
        avg_point.name = "Avg. Supervised OOD"
        avg_lower.name = "Avg. Supervised OOD"
        avg_upper.name = "Avg. Supervised OOD"

        if has_all:
            pn_test_all = (
                pn[pn["test_llm"] == "all"]
                .set_index("train_llm")
                .reindex(col_order_sorted)
            )
            pn_point["all"] = pn_test_all[metric].values
            pn_lower["all"] = pn_test_all[lower_col].values
            pn_upper["all"] = pn_test_all[upper_col].values
            avg_point["all"] = pn_test_all[metric].mean(skipna=True)
            avg_lower["all"] = pn_test_all[lower_col].mean(skipna=True)
            avg_upper["all"] = pn_test_all[upper_col].mean(skipna=True)
            pu_point["all"] = pu_all_point
            pu_lower["all"] = pu_all_lower
            pu_upper["all"] = pu_all_upper

        point_df = pd.concat([pn_point, avg_point.to_frame().T, pu_point])
        lower_df = pd.concat([pn_lower, avg_lower.to_frame().T, pu_lower])
        upper_df = pd.concat([pn_upper, avg_upper.to_frame().T, pu_upper])
        return point_df, lower_df, upper_df

    for metric in metrics:
        point_df, lower_df, upper_df = build_heatmap_df(df, metric, llms_list, ci_level=0.95)

        plot_df = point_df.copy()
        if metric == "bbe":
            plot_df = plot_df - 0.5
            lower_df = lower_df - 0.5
            upper_df = upper_df - 0.5

        # Keep raw data for manual annotation before renaming
        raw_point = plot_df.copy()
        raw_lower = lower_df.copy()
        raw_upper = upper_df.copy()

        label_rename = label_rename_gemini if gemini else label_rename_default
        plot_df = plot_df.rename(index=label_rename, columns=label_rename)

        plt.figure(figsize=(22, 16))

        if metric in binary_metrics or (metric == "tnr" and gemini):
            cmap = "YlOrBr"
            data_min = np.nanmin(plot_df.values)
            data_max = np.nanmax(plot_df.values)
            margin = max((data_max - data_min) * 0.05, 0.01)
            vmin = max(0.0, data_min - margin)
            vmax = min(1.0, data_max + margin)
            center = (vmin + vmax) / 2
        else:
            cmap = orange_white_purple
            if metric == "bbe":
                center = 0.0
                max_dev = np.nanmax(np.abs(plot_df.values - center))
                vmin, vmax = center - max_dev, center + max_dev
            else:
                data_min = np.nanmin(plot_df.values)
                data_max = np.nanmax(plot_df.values)
                margin = max((data_max - data_min) * 0.05, 0.01)
                vmin = max(0.0, data_min - margin)
                vmax = min(1.0, data_max + margin)
                center = 0.5

        ax = sns.heatmap(
            plot_df,
            annot=False,
            fmt="",
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
        )

        # Draw point estimate and CI text separately so they can have different sizes.
        # point_fontsize controls the main number; ci_fontsize controls the [lo, hi] line.
        for i in range(raw_point.shape[0]):
            for j in range(raw_point.shape[1]):
                val = raw_point.iloc[i, j]
                if pd.isna(val):
                    continue
                lo = raw_lower.iloc[i, j]
                hi = raw_upper.iloc[i, j]
                cx, cy = j + 0.5, i + 0.5
                ax.text(cx, cy - 0.15, fmt(val),
                        ha="center", va="center",
                        fontsize=point_fontsize, fontweight="bold",
                        color="black")
                ax.text(cx, cy + 0.28, f"[{fmt(lo)},{fmt(hi)}]",
                        ha="center", va="center",
                        fontsize=ci_fontsize,
                        color="black")
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x))
        )

        _n_total = len(plot_df)
        _n_cols = len(plot_df.columns)
        _sep_y = _n_total - 1
        _gap_h = 0.20
        ax.add_patch(plt.Rectangle(
            (0, _sep_y - _gap_h / 2), _n_cols, _gap_h,
            facecolor="white", edgecolor="none",
            transform=ax.transData, clip_on=True, zorder=3,
        ))
        ax.axhline(y=_sep_y - _gap_h / 2, color="black", linewidth=1, zorder=4)
        ax.axhline(y=_sep_y + _gap_h / 2, color="black", linewidth=1, zorder=4)
        ax.axhline(y=_n_total - 2, color="black", linewidth=4, zorder=4)
        if "all" in llms_list:
            ax.axvline(x=len(plot_df.columns) - 1, color="black", linewidth=4, zorder=4)

        n_total = len(plot_df)
        n_supervised = n_total - 2
        bracket_top = 1.0
        bracket_bot = 1.0 - n_supervised / n_total
        bx = -0.3 if not gemini else -0.38
        tick = 0.025
        for yfrac in [bracket_top, bracket_bot]:
            ax.plot([bx, bx + tick], [yfrac, yfrac],
                    transform=ax.transAxes, clip_on=False, color="black", lw=2.5)
        ax.plot([bx, bx], [bracket_bot, bracket_top],
                transform=ax.transAxes, clip_on=False, color="black", lw=2.5)
        ax.text(bx - 0.03, (bracket_top + bracket_bot) / 2,
                "Supervised\nLearning",
                transform=ax.transAxes, ha="right", va="center", rotation=90,
                fontsize=30, fontweight="bold")

        if title:
            plt.title(name_to_name.get(metric, metric))
        plt.xlabel("Test LLM")
        plt.ylabel("Train LLM / Method")

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/heatmap_{metric}_with_ci.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()


def make_mle_heatmap(df, gemini, title=False):
    llms_list = ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"] if gemini else ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen"]

    metric = "bbe"
    lower_col = f"{metric}_l_0.95"
    upper_col = f"{metric}_u_0.95"

    # -------------------------
    # MLE block
    # -------------------------
    mle = df[df["learning_method"] == "MLE"]

    def pivot_metric(col):
        return (
            mle.pivot(index="train_llm", columns="test_llm", values=col)
            .reindex(index=llms_list, columns=llms_list)
        )

    mle_point = pivot_metric(metric)
    mle_lower = pivot_metric(lower_col)
    mle_upper = pivot_metric(upper_col)

    # -------------------------
    # PU diagonal (same logic as make_heatmap for bbe)
    # -------------------------
    pu = df[df["learning_method"] == "PU"]
    pu_diag = pu[pu["train_llm"] == pu["test_llm"]]

    pu_point = pd.DataFrame(np.nan, index=["PU (diag)"], columns=llms_list)
    pu_lower_df = pu_point.copy()
    pu_upper_df = pu_point.copy()

    for _, row in pu_diag.iterrows():
        llm = row["train_llm"]
        if llm not in llms_list:
            continue
        # Both PN and PU now estimate alpha_human; read directly
        pu_point.loc["PU (diag)", llm] = row["bbe"]
        pu_lower_df.loc["PU (diag)", llm] = row["bbe_l_0.95"]
        pu_upper_df.loc["PU (diag)", llm] = row["bbe_u_0.95"]

    # -------------------------
    # Sort columns by MLE off-diagonal mean (ascending)
    # -------------------------
    mle_no_diag = mle_point.copy()
    np.fill_diagonal(mle_no_diag.values, np.nan)
    avg_point = mle_no_diag.mean(axis=0, skipna=True)
    col_order_sorted = llms_list

    mle_point = mle_point.loc[col_order_sorted, col_order_sorted]
    mle_lower = mle_lower.loc[col_order_sorted, col_order_sorted]
    mle_upper = mle_upper.loc[col_order_sorted, col_order_sorted]
    pu_point = pu_point[col_order_sorted]
    pu_lower_df = pu_lower_df[col_order_sorted]
    pu_upper_df = pu_upper_df[col_order_sorted]

    # -------------------------
    # Average off-diag MLE row
    # -------------------------
    mle_no_diag_sorted = mle_point.copy()
    np.fill_diagonal(mle_no_diag_sorted.values, np.nan)
    mle_no_diag_lower = mle_lower.copy()
    np.fill_diagonal(mle_no_diag_lower.values, np.nan)
    mle_no_diag_upper = mle_upper.copy()
    np.fill_diagonal(mle_no_diag_upper.values, np.nan)

    avg_point = mle_no_diag_sorted.mean(axis=0, skipna=True)
    avg_lower = mle_no_diag_lower.mean(axis=0, skipna=True)
    avg_upper = mle_no_diag_upper.mean(axis=0, skipna=True)

    avg_point.name = "Avg (off-diag)"
    avg_lower.name = "Avg (off-diag)"
    avg_upper.name = "Avg (off-diag)"

    # -------------------------
    # Combine
    # -------------------------
    point_df = pd.concat([mle_point, pu_point, avg_point.to_frame().T])
    lower_df = pd.concat([mle_lower, pu_lower_df, avg_lower.to_frame().T])
    upper_df = pd.concat([mle_upper, pu_upper_df, avg_upper.to_frame().T])

    # BBE shift
    plot_df = point_df - 0.5
    lower_df = lower_df - 0.5
    upper_df = upper_df - 0.5

    # ---- Build annotation matrix
    annot = plot_df.copy().astype(str)
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            val = plot_df.iloc[i, j]
            lo = lower_df.iloc[i, j]
            hi = upper_df.iloc[i, j]
            if pd.isna(val):
                annot.iloc[i, j] = ""
            else:
                annot.iloc[i, j] = f"{fmt(val)}\n[{fmt(lo)}, {fmt(hi)}]" if ci else f"{fmt(val)}"

    label_rename = label_rename_gemini if gemini else label_rename_default
    plot_df = plot_df.rename(index=label_rename, columns=label_rename)
    annot = annot.rename(index=label_rename, columns=label_rename)

    plt.figure(figsize=(18, 20))
    center = 0.0  # bbe is shifted by -0.5, so 0.5 in original space → 0
    max_dev = np.nanmax(np.abs(plot_df.values - center))

    ax = sns.heatmap(
        plot_df,
        annot=annot,
        fmt="",
        cmap=orange_white_purple,
        center=center,
        vmin=center - max_dev,
        vmax=center + max_dev,
        cbar_kws={"label": r'Test $\hat{\alpha}$ (MLE)'},
    )

    # Bold divider between MLE block and the 2 summary rows below
    ax.axhline(y=len(plot_df) - 2, color="black", linewidth=6)

    if title:
        plt.title(name_to_name.get("bbe", "bbe"))
    plt.xlabel("Test LLM")
    plt.ylabel("Train LLM / Method")

    save_folder = output_folder if title else f"{output_folder}/temporal"
    os.makedirs(save_folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        f"{save_folder}/heatmap_mle_bbe_ci.pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.clf()


def make_heatmap_grid(df, metrics, gemini, title=True):
    """Single PDF with one heatmap subplot per metric in metrics."""
    llms_list = (
        ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"]
        if gemini else
        ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen"]
    )
    llms_list += ["all"]

    def build_heatmap_df(df, metric, col_order, ci_level=0.95):
        lower_col = f"{metric}_l_{ci_level}"
        upper_col = f"{metric}_u_{ci_level}"
        col_order_no_all = [c for c in col_order if c != "all"]
        has_all = "all" in col_order

        pn = df[df["learning_method"] == "PN"]

        def pivot_metric(col):
            return (
                pn.pivot(index="train_llm", columns="test_llm", values=col)
                .reindex(index=col_order_no_all, columns=col_order_no_all)
            )

        pn_point = pivot_metric(metric)
        pn_lower = pivot_metric(lower_col)
        pn_upper = pivot_metric(upper_col)

        pu = df[df["learning_method"] == "PU"]
        pu_diag = pu[pu["train_llm"] == pu["test_llm"]]
        pu_point = pd.DataFrame(np.nan, index=["PU + TTA"], columns=col_order_no_all)
        pu_lower = pu_point.copy()
        pu_upper = pu_point.copy()
        pu_all_point = pu_all_lower = pu_all_upper = np.nan

        for _, row in pu_diag.iterrows():
            llm = row["train_llm"]
            if llm == "all" and has_all:
                pu_all_point = row[metric]
                pu_all_lower = row[lower_col]
                pu_all_upper = row[upper_col]
            elif llm in col_order_no_all:
                pu_point.loc["PU + TTA", llm] = row[metric]
                pu_lower.loc["PU + TTA", llm] = row[lower_col]
                pu_upper.loc["PU + TTA", llm] = row[upper_col]

        pn_no_diag_point = pn_point.copy()
        pn_no_diag_lower = pn_lower.copy()
        pn_no_diag_upper = pn_upper.copy()
        np.fill_diagonal(pn_no_diag_point.values, np.nan)
        np.fill_diagonal(pn_no_diag_lower.values, np.nan)
        np.fill_diagonal(pn_no_diag_upper.values, np.nan)

        avg_point = pn_no_diag_point.mean(axis=0, skipna=True)
        avg_lower = pn_no_diag_lower.mean(axis=0, skipna=True)
        avg_upper = pn_no_diag_upper.mean(axis=0, skipna=True)
        col_order_sorted = col_order_no_all

        pn_point = pn_point.loc[col_order_sorted, col_order_sorted]
        pn_lower = pn_lower.loc[col_order_sorted, col_order_sorted]
        pn_upper = pn_upper.loc[col_order_sorted, col_order_sorted]
        pu_point = pu_point[col_order_sorted]
        pu_lower = pu_lower[col_order_sorted]
        pu_upper = pu_upper[col_order_sorted]
        avg_point = avg_point[col_order_sorted]
        avg_lower = avg_lower[col_order_sorted]
        avg_upper = avg_upper[col_order_sorted]
        avg_point.name = "Avg. Supervised OOD"
        avg_lower.name = "Avg. Supervised OOD"
        avg_upper.name = "Avg. Supervised OOD"

        if has_all:
            pn_test_all = (
                pn[pn["test_llm"] == "all"]
                .set_index("train_llm")
                .reindex(col_order_sorted)
            )
            pn_point["all"] = pn_test_all[metric].values
            pn_lower["all"] = pn_test_all[lower_col].values
            pn_upper["all"] = pn_test_all[upper_col].values
            avg_point["all"] = pn_test_all[metric].mean(skipna=True)
            avg_lower["all"] = pn_test_all[lower_col].mean(skipna=True)
            avg_upper["all"] = pn_test_all[upper_col].mean(skipna=True)
            pu_point["all"] = pu_all_point
            pu_lower["all"] = pu_all_lower
            pu_upper["all"] = pu_all_upper

        point_df = pd.concat([pn_point, avg_point.to_frame().T, pu_point])
        lower_df = pd.concat([pn_lower, avg_lower.to_frame().T, pu_lower])
        upper_df = pd.concat([pn_upper, avg_upper.to_frame().T, pu_upper])
        return point_df, lower_df, upper_df

    n = len(metrics)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    label_rename = label_rename_gemini if gemini else label_rename_default

    with matplotlib.rc_context({'font.size': 10+10, 'font.weight': 'bold'}):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 9 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            point_df, lower_df, upper_df = build_heatmap_df(df, metric, llms_list, ci_level=0.95)

            plot_df = point_df.copy()
            if metric in ["bbe", "plugin-int"]:
                plot_df = plot_df - 0.5
                lower_df = lower_df - 0.5
                upper_df = upper_df - 0.5

            annot = plot_df.copy().astype(str)
            for i in range(plot_df.shape[0]):
                for j in range(plot_df.shape[1]):
                    val = plot_df.iloc[i, j]
                    lo = lower_df.iloc[i, j]
                    hi = upper_df.iloc[i, j]
                    annot.iloc[i, j] = "" if pd.isna(val) else (
                        f"{fmt(val)}\n[{fmt(lo)}, {fmt(hi)}]" if ci else f"{fmt(val)}"
                    )

            plot_df_r = plot_df.rename(index=label_rename, columns=label_rename)
            annot_r = annot.rename(index=label_rename, columns=label_rename)

            if metric in binary_metrics or (metric == "tnr" and gemini):
                cmap = "YlOrBr_r" if metric in ("bce", "neg_prob") else "YlOrBr"
                data_min = np.nanmin(plot_df_r.values)
                data_max = np.nanmax(plot_df_r.values)
                margin = max((data_max - data_min) * 0.05, 0.01)
                vmin = max(0.0, data_min - margin)
                vmax = min(1.0, data_max + margin)
                center = (vmin + vmax) / 2
            else:
                cmap = orange_white_purple
                if metric in ["bbe", "plugin-int"]:
                    center = 0.0
                    max_dev = np.nanmax(np.abs(plot_df_r.values - center))
                    vmin, vmax = center - max_dev, center + max_dev
                else:
                    data_min = np.nanmin(plot_df_r.values)
                    data_max = np.nanmax(plot_df_r.values)
                    margin = max((data_max - data_min) * 0.05, 0.01)
                    vmin = max(0.0, data_min - margin)
                    vmax = min(1.0, data_max + margin)
                    center = 0.5

            sns.heatmap(
                plot_df_r, annot=annot_r, fmt="", cmap=cmap,
                center=center, vmin=vmin, vmax=vmax,
                ax=ax, annot_kws={"size": 7+10},
            )
            ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: fmt(x))
            )

            _n_total = len(plot_df_r)
            _n_cols_h = len(plot_df_r.columns)
            _sep_y = _n_total - 1
            _gap_h = 0.20
            ax.add_patch(plt.Rectangle(
                (0, _sep_y - _gap_h / 2), _n_cols_h, _gap_h,
                facecolor="white", edgecolor="none",
                transform=ax.transData, clip_on=True, zorder=3,
            ))
            ax.axhline(y=_sep_y - _gap_h / 2, color="black", linewidth=0.8, zorder=4)
            ax.axhline(y=_sep_y + _gap_h / 2, color="black", linewidth=0.8, zorder=4)
            ax.axhline(y=_n_total - 2, color="black", linewidth=2, zorder=4)
            if "all" in llms_list:
                ax.axvline(x=len(plot_df_r.columns) - 1, color="black", linewidth=2, zorder=4)

            ax.set_title(name_to_name.get(metric, metric), fontsize=12+15, fontweight="bold")
            ax.set_xlabel("Test LLM", fontsize=9+15)
            ax.set_ylabel("Train LLM / Method", fontsize=9+15)
            ax.tick_params(labelsize=7+10)

        for idx in range(len(metrics), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/heatmap_grid.pdf",
            format="pdf", bbox_inches="tight"
        )
        plt.clf()
        plt.close(fig)


if __name__ == "__main__":

    data = pd.read_csv(input_file)
    data = add_accuracy_cols(data)
    data = reverse_bias(reverse_plugin(data))
    # make_heatmap_ci(data, ['tnr'], "gemini" in input_file, title=False)
    # make_heatmap(data, ['tnr'], "gemini" in input_file, title=False)

    # make_mle_heatmap(data, "gemini" in input_file, title=use_title)
    make_heatmap_grid(data, plot_metrics, "gemini" in input_file, title=True)