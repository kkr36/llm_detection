from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os

font = {
    'weight': 'bold',
    'size'  : 35
}
matplotlib.rc('font', **font)

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

input_file = "../logging_accuracy_xy.csv"
output_folder = input_file.split("/")[-1].split(".csv")[0] + "_plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plot_metrics = [
    "auc",
    "pos_prob",
    "neg_prob",
    "entropy_pos",
    "entropy_neg",
    "bbe",
    "plugin",
    "plugin-int"
]

binary_metrics    = ["auc", "pos_prob", "neg_prob", "entropy_pos", "entropy_neg"]
diverging_metrics = ["bbe", "plugin", "plugin-int"]
flip_metrics      = ["pos_prob", "neg_prob", "bbe", "plugin", "plugin-int"]
swap_metrics      = ["entropy_pos", "entropy_neg"]

name_to_name = {
    "auc"        : "AUC",
    "pos_prob"   : "Avg Pred Human",
    "neg_prob"   : "Avg Pred LLM",
    "entropy_pos": "Avg Entropy Human",
    "entropy_neg": "Avg Entropy LLM",
    "bbe"        : r'BiasTest $\hat{\alpha}$',
    "plugin"     : "Bias Plug-In Alpha",
    "plugin-int" : "Bias Avg P(Human)"
}

method_order = ["PN", "TEDn"]


def resolve_cols(metric, ci_level=0.95):
    """Return (point_col, lower_col, upper_col, do_flip).

    Mirrors the transform logic in plot_heatmaps.py:
      - swap_metrics : read the pos<->neg counterpart column, no sign flip
      - flip_metrics : read counterpart column (if applicable) and negate the value
      - everything else: read column as-is
    """
    pos_bool = "pos" in metric
    has_posneg = "pos" in metric or "neg" in metric

    if metric in swap_metrics:
        base = metric.replace("pos", "neg") if pos_bool else metric.replace("neg", "pos")
        return base, f"{base}_l_{ci_level}", f"{base}_u_{ci_level}", False

    elif metric in flip_metrics:
        base = (metric.replace("pos", "neg") if pos_bool else metric.replace("neg", "pos")) if has_posneg else metric
        return base, f"{base}_l_{ci_level}", f"{base}_u_{ci_level}", True

    else:
        return metric, f"{metric}_l_{ci_level}", f"{metric}_u_{ci_level}", False


def make_optimization_heatmap(df, metrics):
    # Average over duplicate (learning_method, train_llm, eval_llm) rows
    key_cols = ["learning_method", "train_llm", "eval_llm"]
    numeric_cols = [c for c in df.columns if c not in key_cols and df[c].dtype != object]
    df = df.groupby(key_cols)[numeric_cols].mean().reset_index()

    # Ordered row labels: (method, train_llm) pairs present in data
    row_keys = [
        (m, l)
        for m in method_order
        for l in sorted(df[df["learning_method"] == m]["train_llm"].unique())
    ]
    row_labels = [f"{m} / {l}" for m, l in row_keys]

    col_order = sorted(df["eval_llm"].unique())

    for metric in metrics:
        point_col, lower_col, upper_col, do_flip = resolve_cols(metric)

        point_rows, lower_rows, upper_rows = [], [], []

        for m, l in row_keys:
            subset = df[(df["learning_method"] == m) & (df["train_llm"] == l)].set_index("eval_llm")

            def get_series(col, sub=subset):
                if col in sub.columns:
                    return sub[col].reindex(col_order)
                return pd.Series(np.nan, index=col_order)

            pt = get_series(point_col)
            lo = get_series(lower_col)
            hi = get_series(upper_col)

            # if do_flip:
            #     pt, lo, hi = 1 - pt, 1 - hi, 1 - lo   # swap lo/hi when negating

            point_rows.append(pt)
            lower_rows.append(lo)
            upper_rows.append(hi)

        point_df = pd.DataFrame(point_rows, index=row_labels, columns=col_order)
        lower_df = pd.DataFrame(lower_rows, index=row_labels, columns=col_order)
        upper_df = pd.DataFrame(upper_rows, index=row_labels, columns=col_order)

        plot_df = point_df.copy()
        if metric == "bbe" or metric == "plugin-int":
            plot_df  = plot_df  - 0.5
            lower_df = lower_df - 0.5
            upper_df = upper_df - 0.5

        # Annotation matrix
        annot = plot_df.copy().astype(str)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                val = plot_df.iloc[i, j]
                lo  = lower_df.iloc[i, j]
                hi  = upper_df.iloc[i, j]
                if pd.isna(val):
                    annot.iloc[i, j] = ""
                else:
                    annot.iloc[i, j] = (
                        f"{fmt(val)}\n[{fmt(lo)}, {fmt(hi)}]" if ci else fmt(val)
                    )

        n_rows = len(row_labels)
        n_cols = len(col_order)
        plt.figure(figsize=(max(10, n_cols * 6), max(8, n_rows * 3)))

        if metric in binary_metrics:
            cmap    = "YlOrBr"
            center  = np.nanmean(plot_df.values)
            max_dev = np.nanmax(np.abs(plot_df.values - center))
            vmin, vmax = center - max_dev, center + max_dev
        else:
            cmap    = orange_white_purple
            center  = 0.0 if metric in ["bbe", "plugin-int", "plugin"] else 0.5
            max_dev = np.nanmax(np.abs(plot_df.values - center))
            vmin, vmax = center - max_dev, center + max_dev

        ax = sns.heatmap(
            plot_df,
            annot=annot,
            fmt="",
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
        )

        # Dividing lines between method groups
        boundary = 0
        for m in method_order[:-1]:
            boundary += sum(1 for mm, _ in row_keys if mm == m)
            ax.axhline(y=boundary, color="black", linewidth=6)

        plt.title(f"{name_to_name.get(metric, metric)} (95% CI)")
        plt.xlabel("Eval Prompt")
        plt.ylabel("Train Method / Prompt")

        plt.tight_layout()
        plt.savefig(
            f"{output_folder}/opt_heatmap_{metric}_ci.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()


if __name__ == "__main__":
    data = pd.read_csv(input_file)
    make_optimization_heatmap(data, plot_metrics)
