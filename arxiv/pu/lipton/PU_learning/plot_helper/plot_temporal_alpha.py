from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

input_file = "../logging_accuracy_temporal_alpha_full_sentence_alpha_temporal.csv"
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

def make_line_plot(metric, x_lab, title, data):

    colors = {
        "PU": "blue",      # matplotlib auto-assigns
        "PN": "red",
    }

    fig, ax = plt.subplots()
    for label, subset in data:
        col = colors[label.split( )[0]]
        linestyle = "-" if "Retrain" in label else ":"
        plt.plot(
                subset[x_lab], 
                subset[metric],                         
                linestyle=linestyle,
                label=label,
                color=col
                )
        ax.fill_between(
            subset[x_lab],
            subset[f"{metric}_l"],
            subset[f"{metric}_u"],
            alpha=0.3,
            color=col
            # label="confidence interval",
        )
    plt.xlabel(x_lab)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{output_folder}/{title}_{metric}.pdf", bbox_inches="tight")
    plt.clf()

def plot_temporal_data(data, metrics):
    # get 2010 data
    rows_2010 = data[(data["train_year"] == 2010) & (data["train_alpha"]==0) & (data['test_alpha']==0.5)]
    pu_2010 = rows_2010[rows_2010["learning_method"]=="PU"]
    pn_2010 = rows_2010[rows_2010["learning_method"]=="PN"]

    rows_retrain_0 = data[(data["train_year"]==data["test_year"]) & (data["train_alpha"]==0)]
    pu_retrain_0 = rows_retrain_0[rows_retrain_0["learning_method"]=="PU"]
    pn_retrain_0 = rows_retrain_0[rows_retrain_0["learning_method"]=="PN"]
    # import pdb; pdb.set_trace()

    for metric in metrics:
        make_line_plot(metric, "test_year", "temporal", [("PU 2010", pu_2010), ("PN 2010", pn_2010), ("PU Retrain", pu_retrain_0), ("PN Retrain", pn_retrain_0)])

def plot_alpha(data, metrics):
    rows = data[data["train_year"] == 2020]
    rows_pu = rows[rows["learning_method"]=="PU"]
    rows_pn = rows[rows["learning_method"]=="PN"]

    for metric in metrics:
        make_line_plot(metric, "train_alpha", "alpha", [("PU Retrain", rows_pu), ("PN Retrain", rows_pn)])

if __name__ == "__main__":

    data = pd.read_csv(input_file)
    plot_temporal_data(data, plot_metrics)
    plot_alpha(data, plot_metrics)