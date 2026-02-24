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

input_file = "../logging_accuracy_llm.csv"
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
    "bbe": r'Test $\hat{\alpha}$ shift',
    "plugin": "Plug-In Alpha",
    "plugin-int": "Avg P(LLM)"
}

def make_heatmap(df, metrics, gemini):
    llms_list = ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"] if gemini else ["Gemini 2.5 Flash", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct"]

    def build_heatmap_df(df, metric, col_order):
        # --- PN block (top rows)
        pn = df[df["learning_method"] == "PN"]
        pn_pivot = (
            pn.pivot(index="train_llm", columns="test_llm", values=metric)
            .reindex(index=col_order, columns=col_order)  # <-- enforce same order
        )

        # --- PU diagonal row (bottom row)
        pu = df[df["learning_method"] == "PU"]
        pu_diag = pu[pu["train_llm"] == pu["test_llm"]]

        pu_row = pd.DataFrame(
            [np.nan] * len(col_order),
            index=col_order,
        ).T
        pu_row.index = ["PU (diag)"]
        pu_row.columns = col_order

        for _, row in pu_diag.iterrows():
            if row["train_llm"] in col_order:
                
                pos_bool = "pos" in metric
                if metric in swap_metrics:
                    to_add = row[metric.replace("pos" if pos_bool else "neg", "neg" if pos_bool else "pos")]
                elif metric in flip_metrics:
                    to_add = 1 - row[metric.replace("pos" if pos_bool else "neg", "neg" if pos_bool else "pos")]
                else:
                    to_add = row[metric]

                pu_row.loc["PU (diag)", row["train_llm"]] = to_add

        # --- Compute average from PN only, excluding diagonal
        pn_no_diag = pn_pivot.copy()
        np.fill_diagonal(pn_no_diag.values, np.nan)

        avg_row = pn_no_diag.mean(axis=0, skipna=True)
        avg_row.name = "Average (off-diag PN)"

        # --- Combine everything
        full_df = pd.concat([pn_pivot, pu_row, avg_row.to_frame().T])
        return full_df
        # import pdb; pdb.set_trace()
        # return pd.concat([pn_pivot, pu_row])


    # -----------------------
    # PLOT ONE HEATMAP PER METRIC
    # -----------------------
    for metric in metrics:
        heatmap_df = build_heatmap_df(df, metric, llms_list)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
        )
        plt.title(metric)
        plt.xlabel("Test LLM")
        plt.ylabel("Train LLM / Method")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/heatmap_{metric}.pdf", format="pdf", bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":

    data = pd.read_csv(input_file)
    make_heatmap(data, plot_metrics, "gemini" in input_file)