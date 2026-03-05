from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18
    }
import matplotlib
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator

input_file = "../logging_accuracy_llm_gemini.csv"
import os
output_folder = input_file.split("/")[-1].split(".csv")[0]
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

flip_metrics = [
    "pos_prob",
    "neg_prob",
    "bbe",
    "plugin",
    "plugin-int"
]

swap_metrics = [
    "entropy_pos",
    "entropy_neg"
]

name_to_name = {
    "train_alpha": r"Train ${\alpha}$",
    "test_year": "Test Year",
    "auc": "AUC",
    "pos_prob": "Avg Pred LLM",
    "neg_prob": "Avg Pred Human",
    "entropy_pos": "Avg Entropy LLM",
    "entropy_neg": "Avg Entropy Human",
    "bbe": r'Test $\hat{\alpha}$',
    "plugin": "Plug-In Alpha",
    "plugin-int": "Avg P(LLM)"
}

def make_heatmap(df, metrics, gemini):
    llms_list = ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"] if gemini else ["Gemini 2.5 Flash", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct"]

    def build_heatmap_df(df, metric, col_order, ci_level=0.95):

        lower_col = f"{metric}_l_{ci_level}"
        upper_col = f"{metric}_u_{ci_level}"

        # -------------------------
        # PN block
        # -------------------------
        pn = df[df["learning_method"] == "PN"]

        def pivot_metric(col):
            return (
                pn.pivot(index="train_llm", columns="test_llm", values=col)
                .reindex(index=col_order, columns=col_order)
            )

        pn_point = pivot_metric(metric)
        pn_lower = pivot_metric(lower_col)
        pn_upper = pivot_metric(upper_col)

        # -------------------------
        # PU diagonal
        # -------------------------
        pu = df[df["learning_method"] == "PU"]
        pu_diag = pu[pu["train_llm"] == pu["test_llm"]]

        pu_point = pd.DataFrame(np.nan, index=["PU (diag)"], columns=col_order)
        pu_lower = pu_point.copy()
        pu_upper = pu_point.copy()

        for _, row in pu_diag.iterrows():
            llm = row["train_llm"]
            if llm not in col_order:
                continue

            pos_bool = "pos" in metric

            # ---------- Helper ----------
            def transform(val, lower=False, upper=False):
                base_metric = metric
                if metric in swap_metrics:
                    base_metric = metric.replace(
                        "pos" if pos_bool else "neg",
                        "neg" if pos_bool else "pos"
                    )
                    return row[
                        f"{base_metric}_{'l' if lower else 'u' if upper else ''}_{ci_level}"
                    ] if (lower or upper) else row[base_metric]

                elif metric in flip_metrics:
                    base_metric = metric.replace(
                        "pos" if pos_bool else "neg",
                        "neg" if pos_bool else "pos"
                    )

                    if lower:
                        return 1 - row[f"{base_metric}_u_{ci_level}"]
                    elif upper:
                        return 1 - row[f"{base_metric}_l_{ci_level}"]
                    else:
                        return 1 - row[base_metric]

                else:
                    if lower:
                        return row[lower_col]
                    elif upper:
                        return row[upper_col]
                    else:
                        return row[metric]

            pu_point.loc["PU (diag)", llm] = transform(row)
            pu_lower.loc["PU (diag)", llm] = transform(row, lower=True)
            pu_upper.loc["PU (diag)", llm] = transform(row, upper=True)

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

        col_order_sorted = avg_point.sort_values(ascending=True).index.tolist()

        pn_point = pn_point[col_order_sorted]
        pn_lower = pn_lower[col_order_sorted]
        pn_upper = pn_upper[col_order_sorted]

        pu_point = pu_point[col_order_sorted]
        pu_lower = pu_lower[col_order_sorted]
        pu_upper = pu_upper[col_order_sorted]

        avg_point = avg_point[col_order_sorted]
        avg_lower = avg_lower[col_order_sorted]
        avg_upper = avg_upper[col_order_sorted]

        avg_point.name = "Average (off-diag PN)"
        avg_lower.name = "Average (off-diag PN)"
        avg_upper.name = "Average (off-diag PN)"

        # -------------------------
        # Combine
        # -------------------------
        point_df = pd.concat([pn_point, pu_point, avg_point.to_frame().T])
        lower_df = pd.concat([pn_lower, pu_lower, avg_lower.to_frame().T])
        upper_df = pd.concat([pn_upper, pu_upper, avg_upper.to_frame().T])

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
                        f"{val:.3f}\n"
                        f"[{lo:.3f}, {hi:.3f}]"
                    )

        plt.figure(figsize=(20, 14))

        mean_val = np.nanmean(plot_df.values)
        max_dev = np.nanmax(np.abs(plot_df.values - mean_val))

        sns.heatmap(
            plot_df,   # <-- use shifted data
            annot=annot,
            fmt="",
            cmap="RdBu",
            center=mean_val,
            vmin=mean_val - max_dev,
            vmax=mean_val + max_dev,
            cbar_kws={"label": name_to_name.get(metric, metric)},
        )

        # sns.heatmap(
        #     point_df,
        #     annot=annot,
        #     fmt="",
        #     cmap="viridis",
        #     cbar_kws={"label": name_to_name.get(metric, metric)},
        # )

        plt.title(f"{name_to_name.get(metric, metric)} (95% CI)")
        plt.xlabel("Test LLM")
        plt.ylabel("Train LLM / Method")

        plt.tight_layout()
        plt.savefig(
            f"{output_folder}/heatmap_{metric}_ci.pdf",
            # f"logging_accuracy_llm_gemini_no_25flash/heatmap_{metric}_ci.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()


if __name__ == "__main__":

    data = pd.read_csv(input_file)
    make_heatmap(data, plot_metrics, "gemini" in input_file)