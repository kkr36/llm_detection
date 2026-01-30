from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20
    }
import matplotlib
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator

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

name_to_name = {
    "train_alpha": r"Train ${\alpha}$",
    "test_year": "Test Year",
    "auc": "AUC",
    "pos_prob": "Avg Pred LLM",
    "neg_prob": "Avg Pred Human",
    "entropy_pos": "Avg Entropy LLM",
    "entropy_neg": "Avg Entropy Human",
    "bbe": r'Test $\hat{\alpha}$ bias',
    "plugin": "Plug-In Alpha",
    "plugin-int": "Avg P(LLM)"
}

def make_line_plot(metric, x_lab, title, data):

    colors = {
        "PU": "orange",      # matplotlib auto-assigns
        "PN": "red",
        "MLE": "green",
        "Plug-In": "blue"
    }

    fig, ax = plt.subplots()
    for label, subset in data:
        col = colors[label.split( )[0]] if "Plug-In" not in label else "blue"
        linestyle = ":" if "2010" in label else "--" if "Alpha" in label else "-" 
        real_metric = metric if "Plug-In" not in label else "plugin-int"
        plt.plot(
                subset[x_lab], 
                subset[real_metric] if metric != "bbe" else subset[real_metric]-.5,                         
                linestyle=linestyle,
                label=label,
                color=col
                )
        ax.fill_between(
            subset[x_lab],
            subset[f"{real_metric}_l"] if metric != "bbe" else subset[f"{real_metric}_l"]-.5,
            subset[f"{real_metric}_u"] if metric != "bbe" else subset[f"{real_metric}_u"]-.5,
            alpha=0.3,
            color=col
            # label="confidence interval",
        )

    if "year" in x_lab:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
    plt.xlabel(name_to_name[x_lab])
    plt.ylabel(name_to_name[metric])
    plt.tight_layout()
    # plt.legend()
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

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
        # removed ("PU 2010", pu_2010), 
        make_line_plot(metric, "test_year", "temporal", [("PN 2010", pn_2010), ("PU Retrain", pu_retrain_0), ("PN Retrain", pn_retrain_0)])

def plot_temporal_alpha(data, metrics):
    # get 2010 data
    rows_2010 = data[(data["train_year"] == 2010) & (data["train_alpha"]==0) & (data['test_alpha']==0.5)]
    pu_2010 = rows_2010[rows_2010["learning_method"]=="PU"]
    pn_2010 = rows_2010[rows_2010["learning_method"]=="PN"]

    # get retrain 0 data
    rows_retrain_0 = data[(data["train_year"]==data["test_year"]) & (data["train_alpha"]==0)]
    pu_retrain_0 = rows_retrain_0[rows_retrain_0["learning_method"]=="PU"]
    pn_retrain_0 = rows_retrain_0[rows_retrain_0["learning_method"]=="PN"]

    # get retrain alpha data
    # Step 1: rows where train_year == test_year
    same_year = data[data["train_year"] == data["test_year"]]

    # Step 2: compute max train_alpha per train_year (within same_year)
    max_alpha_per_year = (
        same_year
        .groupby("train_year")["train_alpha"]
        .transform("max")
    )

    # Step 3: filter rows matching that max
    rows_retrain_alpha = same_year[
        same_year["train_alpha"] == max_alpha_per_year
    ]
    pu_retrain_alpha = rows_retrain_alpha[rows_retrain_alpha["learning_method"]=="PU"]
    pn_retrain_alpha = rows_retrain_alpha[rows_retrain_alpha["learning_method"]=="PN"]
    for metric in metrics:
        # removed ("PU 2010", pu_2010)
        make_line_plot(metric, "test_year", "temporal_alpha", [("PN 2010", pn_2010), ("PU Retrain", pu_retrain_0), ("PN Retrain", pn_retrain_0), ("PU Retrain Alpha", pu_retrain_alpha), ("PN Retrain Alpha", pn_retrain_alpha)])

def plot_alpha(data, metrics):
    rows = data[data["train_year"] == 2020]
    rows_pu = rows[rows["learning_method"]=="PU"]
    rows_pn = rows[rows["learning_method"]=="PN"]

    for metric in metrics:
        make_line_plot(metric, "train_alpha", "alpha", [("PU Retrain", rows_pu), ("PN Retrain", rows_pn)])

def plot_alpha_james(data, metrics):
    rows = data[data["train_year"] == 2020]
    rows_pu = rows[rows["learning_method"]=="PU"]
    rows_pn = rows[rows["learning_method"]=="PN"]
    rows_james = rows[rows["learning_method"]=="MLE"]
    rows_pn = rows[rows["learning_method"]=="PN"]

    for metric in metrics:
        make_line_plot(metric, "train_alpha", "james_alpha", [("PU Retrain (BBE)", rows_pu), ("PN Retrain (BBE)", rows_pn), ("MLE Retrain", rows_james), ("PN (Plug-In)", rows_pn)])

def plot_temporal_james(data, metrics):
    # get 2010 data
    rows_2010 = data[(data["train_year"] == 2010) & (data["train_alpha"]==0) & (data['test_alpha']==0.5)]
    pu_2010 = rows_2010[rows_2010["learning_method"]=="PU"]
    pn_2010 = rows_2010[rows_2010["learning_method"]=="PN"]
    james_2010 = rows_2010[rows_2010["learning_method"]=="MLE"]

    rows_retrain_0 = data[(data["train_year"]==data["test_year"]) & (data["train_alpha"]==0)]
    pu_retrain_0 = rows_retrain_0[rows_retrain_0["learning_method"]=="PU"]
    pn_retrain_0 = rows_retrain_0[rows_retrain_0["learning_method"]=="PN"]
    james_retrain_0 = rows_retrain_0[rows_retrain_0["learning_method"]=="MLE"]
    # import pdb; pdb.set_trace()

    platt_pn_2010 = data[data["learning_method"]=="PN_platt_2010"]

    for metric in metrics:
        # removed ("PU 2010 (BBE)", pu_2010), ("PU Retrain (BBE)", pu_retrain_0)
        make_line_plot(metric, "test_year", "james_temporal", [("PN 2010 (BBE)", pn_2010), ("PN Retrain (BBE)", pn_retrain_0), ("MLE 2010", james_2010), ("MLE Retrain", james_retrain_0), ("PN 2010 Rescaled (Plug-In)", platt_pn_2010)])


if __name__ == "__main__":

    data = pd.read_csv(input_file)
    # plot_temporal_data(data, plot_metrics)
    # plot_alpha(data, plot_metrics)
    # plot_temporal_alpha(data, plot_metrics)

    plot_alpha_james(data, ["bbe"])
    plot_alpha(data, plot_metrics)

    if "combine" not in input_file:
        plot_temporal_james(data, ["bbe"])
        plot_temporal_data(data, plot_metrics)
        plot_temporal_alpha(data, plot_metrics)