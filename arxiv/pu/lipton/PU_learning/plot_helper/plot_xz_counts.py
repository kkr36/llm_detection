from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import os
import math

font = {
    'weight': 'bold',
    'size'  : 35
}
matplotlib.rc('font', **font)

input_file = "../logging_accuracy_xz_counts.csv"
output_folder = input_file.split("/")[-1].split(".csv")[0] + "_paper"
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
    # "tnr",
]

name_to_name = {
    "auc"        : "AUC",
    "accuracy"   : "Bal. Accuracy",
    "pos_prob"   : "Avg. P(human | human)",
    "neg_prob"   : "Avg. P(human | AI)",
    # "entropy_pos": "Shannon Entropy LLM",
    # "entropy_neg": "Shannon Entropy Human",
    # "entropy"    : "Avg. Shannon Entropy",
    "bce"        : "Bal. Cross-Entropy",
    "bbe"        : "Bias",
    # "plugin"     : "Bias Plug-In Alpha",
    "plugin-int" : "Bias Avg P(Human)",
    "tpr"        : "Human Recall",
    "tnr"        : "AI Recall"
}

flip_metrics  = ["pos_prob", "neg_prob", "bbe", "plugin", "plugin-int"]


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
swap_metrics  = ["entropy_pos", "entropy_neg"]
shift_metrics = {"bbe", "plugin-int"}

eval_col_labels = {
    "rewrite_X": "X",
    "rewrite_Z": "Z",
}

method_colors = {
    "PN"  : "orange",
    "TEDn": "purple",
}

eval_llm_colors = {
    "rewrite_X": "green",
    "rewrite_Z": "crimson",
}

# diverging colormap for heatmaps
from matplotlib.colors import LinearSegmentedColormap
orange_white_purple = LinearSegmentedColormap.from_list(
    "orange_white_purple", ["orange", "white", "purple"][::-1]
)


def resolve_cols(metric, ci_level=0.95):
    pos_bool = "pos" in metric
    has_posneg = "pos" in metric or "neg" in metric

    if metric in swap_metrics:
        base = metric.replace("pos", "neg") if pos_bool else metric.replace("neg", "pos")
    elif metric in flip_metrics:
        base = (metric.replace("pos", "neg") if pos_bool else metric.replace("neg", "pos")) if has_posneg else metric
    else:
        base = metric

    return base, f"{base}_l_{ci_level}", f"{base}_u_{ci_level}"


def _apply_shift(metric, pts, los, his):
    if metric in shift_metrics:
        return pts - 0.5, los - 0.5, his - 0.5
    return pts, los, his


def _agg(df, group_cols):
    numeric_cols = [c for c in df.columns if c not in group_cols and df[c].dtype != object]
    return df.groupby(group_cols)[numeric_cols].mean().reset_index()


# ---------------------------------------------------------------------------
# 1. Line plots: x = num_X, separate figure per (method, eval_llm, metric)
# ---------------------------------------------------------------------------
def make_num_x_lineplots(df, metrics, title=False):
    df = df.copy().dropna(subset=["num_X"])
    agg = _agg(df, ["learning_method", "eval_llm", "num_X"]).sort_values("num_X")

    methods   = sorted(df["learning_method"].unique())
    eval_llms = sorted(df["eval_llm"].unique())

    for method in methods:
        for eval_llm in eval_llms:
            sub = agg[(agg["learning_method"] == method) & (agg["eval_llm"] == eval_llm)]
            if sub.empty:
                continue
            x = sub["num_X"].values

            for metric in metrics:
                point_col, lower_col, upper_col = resolve_cols(metric)
                if point_col not in sub.columns:
                    continue

                pts = sub[point_col].values.copy()
                los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                his = sub[upper_col].values.copy() if upper_col in sub.columns else pts
                pts, los, his = _apply_shift(metric, pts, los, his)

                color = method_colors.get(method, "blue")
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(x, pts, marker="o", linewidth=3, markersize=10, color=color,
                        label=f"{method} / eval={eval_col_labels.get(eval_llm, eval_llm)}")
                ax.fill_between(x, los, his, alpha=0.2, color=color)
                ax.set_xlabel("Num X Sentences")
                ax.set_ylabel(name_to_name.get(metric, metric))
                if title:
                    ax.set_title(
                        f"{name_to_name.get(metric, metric)}\n"
                        f"{method}, eval={eval_col_labels.get(eval_llm, eval_llm)}"
                    )
                ax.legend(fontsize=24)
                if metric == "auc":
                    ax.set_ylim(top=1.0)
                save_folder = f"{output_folder}/titled" if title else output_folder
                os.makedirs(save_folder, exist_ok=True)
                plt.tight_layout()
                plt.savefig(
                    f"{save_folder}/numX_{method}_{eval_llm}_{metric}.pdf",
                    format="pdf", bbox_inches="tight"
                )
                plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Line plots: x = num_Z, separate figure per (method, eval_llm, metric)
# ---------------------------------------------------------------------------
def make_num_z_lineplots(df, metrics, title=False):
    df = df.copy().dropna(subset=["num_Z"])
    if df["num_Z"].nunique() < 2:
        return  # num_Z constant in current data – skip
    agg = _agg(df, ["learning_method", "eval_llm", "num_Z"]).sort_values("num_Z")

    methods   = sorted(df["learning_method"].unique())
    eval_llms = sorted(df["eval_llm"].unique())

    for method in methods:
        for eval_llm in eval_llms:
            sub = agg[(agg["learning_method"] == method) & (agg["eval_llm"] == eval_llm)]
            if sub.empty:
                continue
            x = sub["num_Z"].values

            for metric in metrics:
                point_col, lower_col, upper_col = resolve_cols(metric)
                if point_col not in sub.columns:
                    continue

                pts = sub[point_col].values.copy()
                los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                his = sub[upper_col].values.copy() if upper_col in sub.columns else pts
                pts, los, his = _apply_shift(metric, pts, los, his)

                color = method_colors.get(method, "blue")
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(x, pts, marker="o", linewidth=3, markersize=10, color=color,
                        label=f"{method} / eval={eval_col_labels.get(eval_llm, eval_llm)}")
                ax.fill_between(x, los, his, alpha=0.2, color=color)
                ax.set_xlabel("Num Z Sentences")
                ax.set_ylabel(name_to_name.get(metric, metric))
                if title:
                    ax.set_title(
                        f"{name_to_name.get(metric, metric)}\n"
                        f"{method}, eval={eval_col_labels.get(eval_llm, eval_llm)}"
                    )
                ax.legend(fontsize=24)
                if metric == "auc":
                    ax.set_ylim(top=1.0)
                save_folder = f"{output_folder}/titled" if title else output_folder
                os.makedirs(save_folder, exist_ok=True)
                plt.tight_layout()
                plt.savefig(
                    f"{save_folder}/numZ_{method}_{eval_llm}_{metric}.pdf",
                    format="pdf", bbox_inches="tight"
                )
                plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Overlaid line plots: both eval_llms on the same axes, x = num_X
# ---------------------------------------------------------------------------
def make_num_x_lineplots_overlaid(df, metrics, title=False):
    df = df.copy().dropna(subset=["num_X"])
    agg = _agg(df, ["learning_method", "eval_llm", "num_X"]).sort_values("num_X")

    methods   = sorted(df["learning_method"].unique())
    eval_llms = sorted(df["eval_llm"].unique())

    for method in methods:
        for metric in metrics:
            point_col, lower_col, upper_col = resolve_cols(metric)
            fig, ax = plt.subplots(figsize=(12, 7))
            plotted = False

            for eval_llm in eval_llms:
                sub = agg[(agg["learning_method"] == method) & (agg["eval_llm"] == eval_llm)]
                if sub.empty or point_col not in sub.columns:
                    continue

                x   = sub["num_X"].values
                pts = sub[point_col].values.copy()
                los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                his = sub[upper_col].values.copy() if upper_col in sub.columns else pts
                pts, los, his = _apply_shift(metric, pts, los, his)

                color = eval_llm_colors.get(eval_llm, "black")
                ax.plot(x, pts, marker="o", linewidth=3, markersize=10, color=color,
                        label=f"eval={eval_col_labels.get(eval_llm, eval_llm)}")
                ax.fill_between(x, los, his, alpha=0.2, color=color)
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel("Num X Sentences")
            ax.set_ylabel(name_to_name.get(metric, metric))
            if title:
                ax.set_title(f"{name_to_name.get(metric, metric)}\n{method}")
            ax.legend(fontsize=24)
            if metric == "auc":
                ax.set_ylim(top=1.0)
            save_folder = f"{output_folder}/titled" if title else output_folder
            os.makedirs(save_folder, exist_ok=True)
            plt.tight_layout()
            plt.savefig(
                f"{save_folder}/numX_overlaid_{method}_{metric}.pdf",
                format="pdf", bbox_inches="tight"
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Overlaid line plots: both eval_llms on same axes, x = num_Z, num_X fixed at 15000
#    rewrite_X labeled "Naive", rewrite_Z labeled "Adversarial"; legend above figure.
# ---------------------------------------------------------------------------
def make_num_z_lineplots_overlaid(df, metrics, fixed_num_x=15000, title=False):
    df = df.copy().dropna(subset=["num_Z"])
    df = df[df["num_X"] == fixed_num_x]
    if df.empty or df["num_Z"].nunique() < 2:
        return

    agg = _agg(df, ["learning_method", "eval_llm", "num_Z"]).sort_values("num_Z")

    eval_label = {"rewrite_X": "Naive", "rewrite_Z": "Adversarial"}
    methods   = sorted(df["learning_method"].unique())
    eval_llms = sorted(df["eval_llm"].unique())

    for method in methods:
        for metric in metrics:
            point_col, lower_col, upper_col = resolve_cols(metric)
            fig, ax = plt.subplots(figsize=(12, 7))
            plotted = False

            for eval_llm in eval_llms:
                sub = agg[(agg["learning_method"] == method) & (agg["eval_llm"] == eval_llm)]
                if sub.empty or point_col not in sub.columns:
                    continue

                x   = sub["num_Z"].values
                pts = sub[point_col].values.copy()
                los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                his = sub[upper_col].values.copy() if upper_col in sub.columns else pts
                pts, los, his = _apply_shift(metric, pts, los, his)

                color = eval_llm_colors.get(eval_llm, "black")
                label = eval_label.get(eval_llm, eval_llm)
                ax.plot(x, pts, marker="o", linewidth=3, markersize=10, color=color, label=label)
                ax.fill_between(x, los, his, alpha=0.2, color=color)
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel("# Sentences Adversarial")
            ax.set_ylabel(name_to_name.get(metric, metric))
            if title:
                ax.set_title(f"{name_to_name.get(metric, metric)}\n{method}")
            ax.legend(fontsize=24, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
            if metric == "auc":
                ax.set_ylim(top=1.0)
            save_folder = f"{output_folder}/titled" if title else output_folder
            os.makedirs(save_folder, exist_ok=True)
            plt.tight_layout()
            plt.savefig(
                f"{save_folder}/numZ_overlaid_{method}_{metric}.pdf",
                format="pdf", bbox_inches="tight"
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# 6. 2-D heatmap: num_X (rows) × num_Z (cols), one per (method, eval_llm, metric)
#    Falls back gracefully when one dimension has only one unique value.
# ---------------------------------------------------------------------------
def _fmt(v):
    if np.isnan(v):
        return ""
    s = f"{v:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def make_2d_heatmaps(df, metrics, title=False):
    df = df.copy().dropna(subset=["num_X", "num_Z"])
    agg = _agg(df, ["learning_method", "eval_llm", "num_X", "num_Z"])

    nx_vals = sorted(agg["num_X"].unique())
    nz_vals = sorted(agg["num_Z"].unique())

    if len(nx_vals) < 2 and len(nz_vals) < 2:
        print("make_2d_heatmaps: both num_X and num_Z are constant; skipping")
        return

    methods   = sorted(df["learning_method"].unique())
    eval_llms = sorted(df["eval_llm"].unique())

    for method in methods:
        for eval_llm in eval_llms:
            sub = agg[(agg["learning_method"] == method) & (agg["eval_llm"] == eval_llm)]
            if sub.empty:
                continue

            for metric in metrics:
                point_col, lower_col, upper_col = resolve_cols(metric)
                if point_col not in sub.columns:
                    continue

                pivot = sub.pivot(index="num_X", columns="num_Z", values=point_col)
                pivot = pivot.reindex(index=nx_vals, columns=nz_vals)

                plot_df = pivot.copy()
                if metric in shift_metrics:
                    plot_df = plot_df - 0.5

                annot = plot_df.applymap(_fmt)

                binary_metrics = {"auc", "accuracy", "pos_prob", "neg_prob", "entropy_pos", "entropy_neg", "entropy", "bce"}
                if metric in binary_metrics:
                    cmap = "YlOrBr"
                    data_min = np.nanmin(plot_df.values)
                    data_max = np.nanmax(plot_df.values)
                    margin = max((data_max - data_min) * 0.05, 0.01)
                    vmin = max(0.0, data_min - margin)
                    vmax = min(1.0, data_max + margin)
                    center = (vmin + vmax) / 2
                else:
                    cmap = orange_white_purple
                    center = 0.0 if metric in shift_metrics else 0.5
                    max_dev = max(np.nanmax(np.abs(plot_df.values - center)), 0.01)
                    vmin, vmax = center - max_dev, center + max_dev

                fig_w = max(8, 3 * len(nz_vals))
                fig_h = max(6, 2.5 * len(nx_vals))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))

                sns.heatmap(
                    plot_df, annot=annot, fmt="",
                    cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                    ax=ax, linewidths=0.5, linecolor="lightgray",
                )
                ax.set_xlabel("num_Z")
                ax.set_ylabel("num_X")
                if title:
                    ax.set_title(
                        f"{name_to_name.get(metric, metric)}\n"
                        f"{method}, eval={eval_col_labels.get(eval_llm, eval_llm)}"
                    )
                save_folder = f"{output_folder}/titled" if title else output_folder
                os.makedirs(save_folder, exist_ok=True)
                plt.tight_layout()
                plt.savefig(
                    f"{save_folder}/heatmap_{method}_{eval_llm}_{metric}.pdf",
                    format="pdf", bbox_inches="tight"
                )
                plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Faceted strip: one column per eval_llm, rows = metrics, x = num_X
#    Compact overview of all metrics at once.
# ---------------------------------------------------------------------------
def make_faceted_overview(df, metrics, title=False):
    df = df.copy().dropna(subset=["num_X"])
    agg = _agg(df, ["learning_method", "eval_llm", "num_X"]).sort_values("num_X")

    methods   = sorted(df["learning_method"].unique())
    eval_llms = sorted(df["eval_llm"].unique())
    n_metrics = len(metrics)
    n_evals   = len(eval_llms)

    for method in methods:
        fig, axes = plt.subplots(
            n_metrics, n_evals,
            figsize=(9 * n_evals, 6 * n_metrics),
            squeeze=False,
        )

        for row_i, metric in enumerate(metrics):
            point_col, lower_col, upper_col = resolve_cols(metric)

            for col_i, eval_llm in enumerate(eval_llms):
                ax = axes[row_i][col_i]
                sub = agg[(agg["learning_method"] == method) & (agg["eval_llm"] == eval_llm)]
                if sub.empty or point_col not in sub.columns:
                    ax.set_visible(False)
                    continue

                x   = sub["num_X"].values
                pts = sub[point_col].values.copy()
                los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                his = sub[upper_col].values.copy() if upper_col in sub.columns else pts
                pts, los, his = _apply_shift(metric, pts, los, his)

                color = eval_llm_colors.get(eval_llm, "black")
                ax.plot(x, pts, marker="o", linewidth=2.5, markersize=8, color=color)
                ax.fill_between(x, los, his, alpha=0.2, color=color)

                # if row_i == 0:
                    # ax.set_title(f"eval={eval_col_labels.get(eval_llm, eval_llm)}", fontsize=28)
                if col_i == 0:
                    ax.set_ylabel(name_to_name.get(metric, metric), fontsize=22)
                if row_i == n_metrics - 1:
                    ax.set_xlabel("Num X Sentences", fontsize=22)
                ax.tick_params(labelsize=18)

        if title:
            fig.suptitle(f"All Metrics — {method}", fontsize=36, fontweight="bold", y=1.01)
        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/faceted_overview_{method}.pdf",
            format="pdf", bbox_inches="tight"
        )
        plt.close(fig)


def make_num_z_lineplots_overlaid_grid(df, metrics, fixed_num_x=15000, title=True):
    from matplotlib.ticker import MaxNLocator

    """One PDF per method: grid of subplots, one subplot per metric, both eval_llms overlaid."""
    df = df.copy().dropna(subset=["num_Z"])
    df = df[df["num_X"] == fixed_num_x]
    if df.empty or df["num_Z"].nunique() < 2:
        return

    agg = _agg(df, ["learning_method", "eval_llm", "num_Z"]).sort_values("num_Z")
    eval_label = {"rewrite_X": "Naive", "rewrite_Z": "Adversarial"}
    methods   = sorted(df["learning_method"].unique())
    eval_llms = sorted(df["eval_llm"].unique())

    n = len(metrics)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    for method in methods:
        with matplotlib.rc_context({"font.size": 25, "font.weight": "bold"}):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
            axes_flat = axes.flatten()

            for idx, metric in enumerate(metrics):
                ax = axes_flat[idx]
                point_col, lower_col, upper_col = resolve_cols(metric)
                plotted = False

                for eval_llm in eval_llms:
                    sub = agg[(agg["learning_method"] == method) & (agg["eval_llm"] == eval_llm)]
                    if sub.empty or point_col not in sub.columns:
                        continue

                    x   = sub["num_Z"].values
                    pts = sub[point_col].values.copy()
                    los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                    his = sub[upper_col].values.copy() if upper_col in sub.columns else pts
                    pts, los, his = _apply_shift(metric, pts, los, his)

                    color = eval_llm_colors.get(eval_llm, "black")
                    label = eval_label.get(eval_llm, eval_llm)
                    ax.plot(x, pts, marker="o", linewidth=2.5, markersize=8, color=color, label=label)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                    ax.fill_between(x, los, his, alpha=0.2, color=color)
                    plotted = True

                if not plotted:
                    ax.set_visible(False)
                    continue

                ax.set_xlabel("# Sentences Adversarial", fontsize=25)
                ax.set_ylabel(name_to_name.get(metric, metric), fontsize=25)
                ax.set_title(name_to_name.get(metric, metric), fontsize=25, fontweight="bold")
                ax.tick_params(labelsize=20)
                if metric == "auc":
                    ax.set_ylim(top=1.0)

            handles, labels_leg = axes_flat[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels_leg, loc="upper center", ncol=len(eval_llms),
                           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=25)

            for idx in range(len(metrics), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            if title:
                fig.suptitle(f"# Sentences Adversarial — {method}", fontsize=28, fontweight="bold", y=1.04)

            save_folder = f"{output_folder}/titled" if title else output_folder
            os.makedirs(save_folder, exist_ok=True)
            plt.tight_layout()
            plt.savefig(
                f"{save_folder}/numZ_overlaid_grid_{method}.pdf",
                format="pdf", bbox_inches="tight"
            )
            plt.clf()
            plt.close(fig)


if __name__ == "__main__":
    data = pd.read_csv(input_file)
    data = add_accuracy_cols(data)
    data["pos_prob"] = 1-data["pos_prob"]
    for ul in ["u", "l"]:
        for ci in ["0.9", "0.95", "0.99"]:
            format_str = f"{ul}_{ci}"
            data[f"pos_prob_{format_str}"] = 1-data[f"pos_prob_{format_str}"]

    for use_title in [False, True][:1]:
        # make_num_x_lineplots(data, plot_metrics, title=use_title)
        # make_num_z_lineplots(data, plot_metrics, title=use_title)
        # make_num_x_lineplots_overlaid(data, plot_metrics, title=use_title)
        make_num_z_lineplots_overlaid_grid(data, plot_metrics, title=use_title)
        # make_num_z_lineplots_overlaid(data, plot_metrics, title=use_title)
        # make_2d_heatmaps(data, plot_metrics, title=use_title)
        # make_faceted_overview(data, plot_metrics, title=use_title)
