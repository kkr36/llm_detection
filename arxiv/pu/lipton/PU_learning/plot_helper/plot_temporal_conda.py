"""Temporal plot with ConDA overlaid on the PNU/Supervised/Liang comparison.

Sibling of plot_temporal_pnu.py (which stays untouched). It reads BOTH:
  * ../logging_accuracy_temporal_alpha_full_sentence_temporal.csv        (PN/TEDn/PNU/MLE)
  * ../logging_accuracy_temporal_alpha_full_sentence_conda_temporal.csv  (ConDA)
concatenates them, and draws an added "ConDA (DA)" curve (dash-dot, purple)
alongside Supervised-2010 / PU + TTA / PNU + TTA / Liang.

Outputs go to a NEW folder so the existing *_PNU plots are not overwritten.
"""
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20
    }
import matplotlib
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator

# The PNU/Supervised/Liang comparison CSV and the new ConDA CSV are concatenated.
input_file = "../logging_accuracy_temporal_alpha_full_sentence_temporal.csv"
conda_input_file = "../logging_accuracy_temporal_alpha_full_sentence_conda_temporal.csv"
import os
# New output folder keyed off the ConDA CSV so the existing *_PNU plots stay put.
output_folder = conda_input_file.split("/")[-1].split(".csv")[0] + "_PNU"
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

name_to_name = {
    "auc"        : "AUC",
    "accuracy"   : "Bal. Accuracy",
    "pos_prob"   : "Avg. P(human | AI)",
    "neg_prob"   : "Avg. P(human | human)",
    "entropy_pos": "Shannon Entropy Human",
    "entropy_neg": "Shannon Entropy LLM",
    "entropy"    : "Avg. Shannon Entropy",
    "bce"        : "Bal. Cross-Entropy",
    "bbe"        : "Bias",
    # "plugin"     : "Bias Plug-In Alpha",
    "plugin-int" : "Bias Avg P(AI) wrt 2010",
    "tpr"        : "AI Recall",
    "tnr"        : "Human Recall",
    "test_year": "Test Year",
}

# Color for the added ConDA series; shared by every plotting helper below.
CONDA_COLOR = "rebeccapurple"

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

def reverse_pos(df, ci_level=0.95):
    """Balanced accuracy = (TPR + TNR) / 2. In this file's CSV convention,
    pos_prob = TPR (Avg Pred Human) and neg_prob = FPR (Avg Pred LLM)."""
    df = df.copy()
    ci = str(ci_level)
    df["pos_prob"]              = 1-df["pos_prob"]
    df[f"pos_prob_l_{ci}"]     = 1-df[f"pos_prob_l_{ci}"]
    df[f"pos_prob_u_{ci}"]     = 1-df[f"pos_prob_u_{ci}"]
    return df

def reverse_neg(df, ci_level=0.95):
    """Balanced accuracy = (TPR + TNR) / 2. In this file's CSV convention,
    pos_prob = TPR (Avg Pred Human) and neg_prob = FPR (Avg Pred LLM)."""
    df = df.copy()
    ci = str(ci_level)
    df["neg_prob"]              = 1-df["neg_prob"]
    df[f"neg_prob_l_{ci}"]     = 1-df[f"neg_prob_l_{ci}"]
    df[f"neg_prob_u_{ci}"]     = 1-df[f"neg_prob_u_{ci}"]
    return df

def _series_color(label, colors):
    """Color lookup that also handles the added ConDA series."""
    if "ConDA" in label:
        return CONDA_COLOR
    if "Optimal" in label:
        return "black"
    if "Plug-In" in label:
        return "darkorange"
    return colors[label.split()[0]]

def _series_linestyle(label):
    """Dash-dot for ConDA, dashed for a frozen-2010 model, dotted for Alpha, else solid."""
    if "ConDA" in label:
        return "-."
    if "2010" in label:
        return "--"
    if "Alpha" in label:
        return ":"
    return "-"

def make_line_plot(metric, x_lab, title, data, show_title=False):

    colors = {
        "PU": "steelblue",
        "PNU": "forestgreen",
        "Supervised": "red",
        "Liang": "burlywood",
        "Plug-In": "darkorange",
        "Optimal": "black",
        "ConDA": CONDA_COLOR,
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, subset in data:
        subset = subset.sort_values(x_lab)
        col = _series_color(label, colors)
        linestyle = _series_linestyle(label)
        real_metric = metric if "Plug-In" not in label else "plugin-int"
        try:
            y_plot = subset[real_metric] if metric not in ["bbe", "plugin-int"] else subset[real_metric]-subset[real_metric].tolist()[0]
        except:
            import pdb; pdb.set_trace()
        y_upper = subset[f"{real_metric}_l_0.95"] if metric not in ["bbe", "plugin-int"] else subset[f"{real_metric}_l_0.95"]-subset[real_metric].tolist()[0]
        y_lower = subset[f"{real_metric}_u_0.95"] if metric not in ["bbe", "plugin-int"] else subset[f"{real_metric}_u_0.95"]-subset[real_metric].tolist()[0]

        plt.plot(
                subset[x_lab],
                y_plot,
                linestyle=linestyle,
                label=label,
                color=col,
                linewidth=2.5
                )
        ax.fill_between(
            subset[x_lab],
            y_upper,
            y_lower,
            alpha=0.2,
            color=col
        )

    if "year" in x_lab:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
    plt.xlabel(name_to_name[x_lab])
    ylab_extra = f", wrt {2010 if 'temporal' in title else 0}"
    plt.ylabel(name_to_name[metric] + (ylab_extra if metric=="bbe" else ''))
    if show_title:
        plt.title(name_to_name[metric])
    plt.tight_layout()

    save_folder = f"{output_folder}/titled" if show_title else output_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f"{save_folder}/{title}_{metric}.pdf", bbox_inches="tight")
    plt.clf()

def plot_temporal_alpha(data, metrics, show_title=False):
    rows_2010 = data[(data["train_year"] == 2010) & (data["train_alpha"]==0) & (data['test_alpha']==0.5)]
    pn_2010 = rows_2010[rows_2010["learning_method"]=="PN"]

    rows_retrain = data[data["train_year"]==data["test_year"]]
    tedn_retrain = rows_retrain[rows_retrain["learning_method"]=="TEDn"]
    pnu_retrain = rows_retrain[rows_retrain["learning_method"]=="PNU"]
    conda_retrain = rows_retrain[rows_retrain["learning_method"]=="ConDA"]

    for metric in metrics:
        make_line_plot(metric, "test_year", "temporal_alpha", [("Supervised 2010", pn_2010), ("PU Retrain", tedn_retrain), ("PNU + TTA", pnu_retrain), ("ConDA (DA)", conda_retrain)], show_title=show_title)


def plot_temporal_combined(data, show_title=False):
    from matplotlib.lines import Line2D

    rows_2010 = data[(data["train_year"] == 2010) & (data["train_alpha"]==0) & (data['test_alpha']==0.5)]
    pn_2010 = rows_2010[rows_2010["learning_method"]=="PN"]
    james_2010 = rows_2010[rows_2010["learning_method"]=="MLE"]

    rows_retrain_alpha = data[data["train_year"]==data["test_year"]]
    tedn_retrain = rows_retrain_alpha[rows_retrain_alpha["learning_method"]=="TEDn"]
    pnu_retrain = rows_retrain_alpha[rows_retrain_alpha["learning_method"]=="PNU"]
    conda_retrain = rows_retrain_alpha[rows_retrain_alpha["learning_method"]=="ConDA"]

    rows_retrain_pu = data[(data["train_year"]==data["test_year"]) & (data["train_alpha"]==0.5)]
    pu_retrain_0 = rows_retrain_pu[rows_retrain_pu["learning_method"]=="TEDn"]
    pnu_retrain_0 = rows_retrain_pu[rows_retrain_pu["learning_method"]=="PNU"]
    conda_retrain_0 = rows_retrain_pu[rows_retrain_pu["learning_method"]=="ConDA"]

    colors = {"PU": "steelblue", "PNU": "forestgreen", "Supervised": "red", "Liang": "burlywood", "ConDA": CONDA_COLOR}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: tnr (mirrors plot_temporal_alpha)
    left_stat = "tnr"
    for label, subset in [("Supervised 2010", pn_2010), ("PU with Test-Time Adaptation", tedn_retrain), ("PNU + TTA", pnu_retrain), ("ConDA (DA)", conda_retrain)]:
        subset = subset.sort_values("test_year")
        col = _series_color(label, colors)
        linestyle = _series_linestyle(label)
        y_plot = subset[left_stat]
        y_upper = subset[f"{left_stat}_l_0.95"]
        y_lower = subset[f"{left_stat}_u_0.95"]
        ax_left.plot(subset["test_year"], y_plot, linestyle=linestyle, color=col, linewidth=2.5)
        ax_left.fill_between(subset["test_year"], y_upper, y_lower, alpha=0.2, color=col)
    ax_left.set_xticks([2010, 2015, 2020])
    ax_left.set_xlabel(name_to_name["test_year"])
    ax_left.set_ylabel(name_to_name[left_stat])

    # Right panel: bbe (mirrors plot_temporal_james)
    for label, subset in [("Supervised 2010", pn_2010), ("PU with Test-Time Adaptation", pu_retrain_0), ("PNU + TTA", pnu_retrain_0), ("ConDA (DA)", conda_retrain_0), ("Liang et al 2010", james_2010)]:
        subset = subset.sort_values("test_year")
        col = _series_color(label, colors)
        linestyle = _series_linestyle(label)
        if len(subset) == 0:
            continue
        baseline = subset["bbe"].tolist()[0]
        y_plot = subset["bbe"] - baseline
        y_upper = subset["bbe_l_0.95"] - baseline
        y_lower = subset["bbe_u_0.95"] - baseline
        ax_right.plot(subset["test_year"], y_plot, linestyle=linestyle, color=col, linewidth=2.5)
        ax_right.fill_between(subset["test_year"], y_upper, y_lower, alpha=0.2, color=col)
    ax_right.axhline(y=0, color="gray", linestyle=":", linewidth=1.5, zorder=0)
    ax_right.set_xticks([2010, 2015, 2020])
    ax_right.set_xlabel(name_to_name["test_year"])
    ax_right.set_ylabel(name_to_name["bbe"] + " wrt 2010")

    legend_handles = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, label='Supervised 2010'),
        Line2D([0], [0], color='burlywood', linestyle='--', linewidth=2.5, label='Liang et al. 2010'),
        Line2D([0], [0], color='steelblue', linestyle='-', linewidth=2.5, label='PU + TTA'),
        Line2D([0], [0], color='forestgreen', linestyle='-', linewidth=2.5, label='PNU + TTA'),
        Line2D([0], [0], color=CONDA_COLOR, linestyle='-.', linewidth=2.5, label='ConDA (DA)'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25), frameon=False)

    if show_title:
        fig.suptitle(f"{name_to_name[left_stat]} & {name_to_name['bbe']}", fontweight="bold")

    plt.tight_layout()
    save_folder = f"{output_folder}/titled" if show_title else output_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f"{save_folder}/combined_{left_stat}_bbe.pdf", bbox_inches="tight")
    plt.clf()


def plot_temporal_alpha_grid(data, metrics, show_title=True):
    """Single PDF with one subplot per metric for the temporal_alpha line plots."""
    rows_2010 = data[(data["train_year"] == 2010) & (data["train_alpha"] == 0) & (data["test_alpha"] == 0.5)]
    pn_2010 = rows_2010[rows_2010["learning_method"] == "PN"]
    rows_retrain = data[data["train_year"] == data["test_year"]]
    tedn_retrain = rows_retrain[rows_retrain["learning_method"] == "TEDn"]
    pnu_retrain = rows_retrain[rows_retrain["learning_method"] == "PNU"]
    conda_retrain = rows_retrain[rows_retrain["learning_method"] == "ConDA"]

    series = [("Supervised 2010", pn_2010), ("PU + TTA", tedn_retrain), ("PNU + TTA", pnu_retrain), ("ConDA (DA)", conda_retrain)]
    x_lab = "test_year"

    colors = {
        "PU": "steelblue",
        "PNU": "forestgreen",
        "Supervised": "red",
        "Liang": "burlywood",
        "Plug-In": "darkorange",
        "Optimal": "black",
        "ConDA": CONDA_COLOR,
    }

    n = len(metrics)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    with matplotlib.rc_context({"font.size": 20, "font.weight": "bold"}):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]

            for label, subset in series:
                subset = subset.sort_values(x_lab)
                col = _series_color(label, colors)
                linestyle = _series_linestyle(label)
                real_metric = metric if "Plug-In" not in label else "plugin-int"
                try:
                    y_plot = subset[real_metric] if metric not in ["bbe", "plugin-int"] else subset[real_metric] - subset[real_metric].tolist()[0]
                    y_upper = subset[f"{real_metric}_l_0.95"] if metric not in ["bbe", "plugin-int"] else subset[f"{real_metric}_l_0.95"] - subset[real_metric].tolist()[0]
                    y_lower = subset[f"{real_metric}_u_0.95"] if metric not in ["bbe", "plugin-int"] else subset[f"{real_metric}_u_0.95"] - subset[real_metric].tolist()[0]
                    if real_metric == "pos_prob":
                        y_upper = 1 - y_upper
                        y_lower = 1 - y_lower
                    if real_metric == "neg_prob":
                        y_plot = 1 - y_plot
                except Exception:
                    continue
                ax.plot(subset[x_lab], y_plot, linestyle=linestyle, label=label, color=col, linewidth=2)
                ax.fill_between(subset[x_lab], y_upper, y_lower, alpha=0.2, color=col)

            if "year" in x_lab:
                ax.set_xticks([2010, 2015, 2020])
            ax.set_xlabel(name_to_name[x_lab], fontsize=19)
            ylab_extra = ", wrt 2010"
            ax.set_ylabel(name_to_name[metric] + (ylab_extra if metric == "bbe" else ""), fontsize=19)
            ax.set_title(name_to_name.get(metric, metric), fontsize=22, fontweight="bold")
            ax.tick_params(labelsize=17)

        handles, labels_leg = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels_leg, loc="upper center", ncol=len(series),
                       bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=20)

        for idx in range(len(metrics), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        save_folder = f"{output_folder}/titled" if show_title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/temporal_alpha_grid.pdf",
            format="pdf", bbox_inches="tight"
        )
        plt.clf()
        plt.close(fig)


if __name__ == "__main__":
    data = pd.read_csv(input_file)
    conda_data = pd.read_csv(conda_input_file)
    data = pd.concat([data, conda_data], ignore_index=True)

    data = add_accuracy_cols(data)
    data = reverse_neg(reverse_pos(data))
    data["neg_prob"] = 1 - data["neg_prob"]
    for ul in ["u", "l"]:
        for ci in ["0.9", "0.95", "0.99"]:
            format_str = f"{ul}_{ci}"
            data[f"pos_prob_{format_str}"] = 1-data[f"pos_prob_{format_str}"]

    plot_temporal_combined(data, show_title=False)
    plot_temporal_alpha_grid(data, plot_metrics, False)
