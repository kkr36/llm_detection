from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

font = {
    'weight': 'bold',
    'size'  : 35
}
matplotlib.rc('font', **font)

import os

input_file = "../logging_accuracy_xz_frac.csv"
output_folder = input_file.split("/")[-1].split(".csv")[0] + "_paper"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plot_metrics = [
    "auc",
    "accuracy",
    "pos_prob",
    "neg_prob",
    "entropy_pos",
    "entropy_neg",
    "entropy",
    "bce",
    "bbe",
    "plugin",
    "plugin-int",
]

flip_metrics = ["pos_prob", "neg_prob", "bbe", "plugin", "plugin-int"]
swap_metrics = ["entropy_pos", "entropy_neg"]

name_to_name = {
    "auc"        : "AUC",
    "accuracy"   : "Accuracy",
    "pos_prob"   : "Avg P(human | human)",
    "neg_prob"   : "Avg P(human | LLM)",
    "entropy_pos": "Avg Entropy Human",
    "entropy_neg": "Avg Entropy LLM",
    "entropy"    : "Avg Binary Entropy",
    "bce"        : "Balanced Cross-Entropy",
    "bbe"        : r'BiasTest $\hat{\alpha}$',
    "plugin"     : "Bias Plug-In Alpha",
    "plugin-int" : "Bias Avg P(Human)"
}

eval_col_labels = {
    "rewrite_X": "X",
    "rewrite_Z": "Z",
}

method_colors = {
    "PN"  : "orange",
    "TEDn": "purple",
}

eval_llm_colors = {
    "rewrite_X": "steelblue",
    "rewrite_Z": "crimson",
}


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

def resolve_cols(metric, ci_level=0.95):
    """Return (point_col, lower_col, upper_col)."""
    pos_bool = "pos" in metric
    has_posneg = "pos" in metric or "neg" in metric

    if metric in swap_metrics:
        base = metric.replace("pos", "neg") if pos_bool else metric.replace("neg", "pos")
    elif metric in flip_metrics:
        base = (metric.replace("pos", "neg") if pos_bool else metric.replace("neg", "pos")) if has_posneg else metric
    else:
        base = metric

    return base, f"{base}_l_{ci_level}", f"{base}_u_{ci_level}"


def make_frac_lineplots(df, metrics, title=False):
    """
    For each (learning_method, eval_llm, metric) triple, produce one line plot:
      x-axis = fraction of prompt Z (float extracted from train_llm, e.g. xz_0.3 -> 0.3)
      y-axis = metric value (with same offset logic as plot_optimization_xz)
      x-label = "Fraction Prompt Z"
    """
    # Parse the fraction from train_llm (format: "xz_<float>")
    df = df.copy()
    df["frac_z"] = df["train_llm"].str.replace("xz_", "", regex=False).astype(float)

    key_cols = ["learning_method", "eval_llm", "frac_z"]
    numeric_cols = [c for c in df.columns if c not in key_cols and df[c].dtype != object]
    df_agg = df.groupby(key_cols)[numeric_cols].mean().reset_index()
    df_agg = df_agg.sort_values("frac_z")

    methods   = ["PN", "TEDn"]
    eval_llms = ["rewrite_X", "rewrite_Z"]

    for method in methods:
        for eval_llm in eval_llms:
            sub = df_agg[
                (df_agg["learning_method"] == method) &
                (df_agg["eval_llm"] == eval_llm)
            ]
            if sub.empty:
                continue

            x = sub["frac_z"].values

            for metric in metrics:
                point_col, lower_col, upper_col = resolve_cols(metric)

                if point_col not in sub.columns:
                    continue

                pts = sub[point_col].values.copy()
                los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                his = sub[upper_col].values.copy() if upper_col in sub.columns else pts

                if metric in ("bbe", "plugin-int"):
                    pts -= 0.5
                    los -= 0.5
                    his -= 0.5

                color = method_colors.get(method, "blue")
                label = f"{method} / eval={eval_col_labels.get(eval_llm, eval_llm)}"

                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(x, pts, marker="o", linewidth=3, markersize=10,
                        label=label, color=color)
                ax.fill_between(x, los, his, alpha=0.2, color=color)

                ax.set_xlabel("Fraction Prompt Z")
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
                fname = f"{save_folder}/frac_{method}_{eval_llm}_{metric}.pdf"
                plt.savefig(fname, format="pdf", bbox_inches="tight")
                plt.clf()
                plt.close(fig)


def make_num_lineplots(df, metrics, title=False):
    """
    Identical to make_frac_lineplots but x-axis is num_Z (for eval=rewrite_Z plots)
    or num_X (for eval=rewrite_X plots) — the actual sentence count in the training
    data rather than the fraction.
    """
    df = df.copy()

    # Drop rows where num_X / num_Z are missing (script not yet run)
    if "num_X" not in df.columns or "num_Z" not in df.columns:
        print("WARNING: num_X / num_Z columns not found; skipping make_num_lineplots")
        return

    df = df.dropna(subset=["num_X", "num_Z"])
    if df.empty:
        return

    # x-axis value per eval_llm: num_Z when evaluating on Z, num_X when evaluating on X
    def pick_num_col(eval_llm):
        return "num_Z" if eval_llm == "rewrite_Z" else "num_X"

    key_cols = ["learning_method", "eval_llm", "num_X", "num_Z"]
    numeric_cols = [c for c in df.columns if c not in key_cols and df[c].dtype != object]
    df_agg = df.groupby(key_cols)[numeric_cols].mean().reset_index()

    methods   = ["PN", "TEDn"]
    eval_llms = ["rewrite_X", "rewrite_Z"]

    for method in methods:
        for eval_llm in eval_llms:
            num_col = pick_num_col(eval_llm)
            sub = df_agg[
                (df_agg["learning_method"] == method) &
                (df_agg["eval_llm"] == eval_llm)
            ].sort_values(num_col)
            if sub.empty:
                continue

            x = sub[num_col].values

            for metric in metrics:
                point_col, lower_col, upper_col = resolve_cols(metric)

                if point_col not in sub.columns:
                    continue

                pts = sub[point_col].values.copy()
                los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                his = sub[upper_col].values.copy() if upper_col in sub.columns else pts

                if metric in ("bbe", "plugin-int"):
                    pts -= 0.5
                    los -= 0.5
                    his -= 0.5

                color = method_colors.get(method, "blue")
                label = f"{method} / eval={eval_col_labels.get(eval_llm, eval_llm)}"
                x_label = f"Num Prompt {'Z' if eval_llm == 'rewrite_Z' else 'X'} Sentences"

                fig, ax = plt.subplots(figsize=(12, 7))
                ax.plot(x, pts, marker="o", linewidth=3, markersize=10,
                        label=label, color=color)
                ax.fill_between(x, los, his, alpha=0.2, color=color)

                ax.set_xlabel(x_label)
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
                fname = f"{save_folder}/num_{method}_{eval_llm}_{metric}.pdf"
                plt.savefig(fname, format="pdf", bbox_inches="tight")
                plt.clf()
                plt.close(fig)


def make_frac_lineplots_overlaid(df, metrics, title=False):
    """
    For each (learning_method, metric) pair, produce one plot with rewrite_X
    and rewrite_Z curves overlaid.
      x-axis = fraction of prompt Z
      x-label = "Fraction Prompt Z"
    """
    df = df.copy()
    df["frac_z"] = df["train_llm"].str.replace("xz_", "", regex=False).astype(float)

    key_cols = ["learning_method", "eval_llm", "frac_z"]
    numeric_cols = [c for c in df.columns if c not in key_cols and df[c].dtype != object]
    df_agg = df.groupby(key_cols)[numeric_cols].mean().reset_index()
    df_agg = df_agg.sort_values("frac_z")

    methods   = ["PN", "TEDn"]
    eval_llms = ["rewrite_X", "rewrite_Z"]

    for method in methods:
        for metric in metrics:
            point_col, lower_col, upper_col = resolve_cols(metric)

            fig, ax = plt.subplots(figsize=(12, 7))

            for eval_llm in eval_llms:
                sub = df_agg[
                    (df_agg["learning_method"] == method) &
                    (df_agg["eval_llm"] == eval_llm)
                ]
                if sub.empty or point_col not in sub.columns:
                    continue

                x   = sub["frac_z"].values
                pts = sub[point_col].values.copy()
                los = sub[lower_col].values.copy() if lower_col in sub.columns else pts
                his = sub[upper_col].values.copy() if upper_col in sub.columns else pts

                if metric in ("bbe", "plugin-int"):
                    pts -= 0.5
                    los -= 0.5
                    his -= 0.5

                color = eval_llm_colors.get(eval_llm, "black")
                label = f"eval={eval_col_labels.get(eval_llm, eval_llm)}"
                ax.plot(x, pts, marker="o", linewidth=3, markersize=10,
                        label=label, color=color)
                ax.fill_between(x, los, his, alpha=0.2, color=color)

            ax.set_xlabel("Fraction Prompt Z")
            ax.set_ylabel(name_to_name.get(metric, metric))
            if title:
                ax.set_title(f"{name_to_name.get(metric, metric)}\n{method}")
            ax.legend(fontsize=24)
            if metric == "auc":
                ax.set_ylim(top=1.0)

            save_folder = f"{output_folder}/titled" if title else output_folder
            os.makedirs(save_folder, exist_ok=True)
            plt.tight_layout()
            fname = f"{save_folder}/frac_overlaid_{method}_{metric}.pdf"
            plt.savefig(fname, format="pdf", bbox_inches="tight")
            plt.clf()
            plt.close(fig)


if __name__ == "__main__":
    data = pd.read_csv(input_file)
    data = add_accuracy_cols(data)
    for use_title in [False, True]:
        make_frac_lineplots(data, plot_metrics, title=use_title)
        make_num_lineplots(data, plot_metrics, title=use_title)
        make_frac_lineplots_overlaid(data, plot_metrics, title=use_title)
