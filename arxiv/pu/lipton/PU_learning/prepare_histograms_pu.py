import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import torch

from model_inference import get_preds, get_u_data
from model_helper import *

font = {'weight': 'bold', 'size': 20}
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator

# SWITCHES
entrance_path = "logging_accuracy_temporal_alpha_full_sentence"
data_type = "ArXiv_BERT"
flip = False
combine = False
sentence = True
clean = True
platt = False
gemini = False
add = False
epochs = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][:]

# train_years = [2010, 2012, 2014, 2016, 2018, 2020]
train_years = [2010, 2016, 2020]
alpha = 0.5

# year -> (probs, targets) for the PU/TEDn model trained+tested on that year
year_dict = {}

for train_year in train_years:
    all_probs = []
    all_targets = []

    for seed in seeds:
        alpha_dir = Path(f"{entrance_path}/{'sentence' if sentence else 'abstract'}_{train_year}/{alpha}_{seed}/ArXiv_BERT_{epochs}")
        pts_pu = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "TEDn" in p.name][0]

        net = get_model("DistilBert")
        state_dict = torch.load(pts_pu, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)

        print(f"train: PU TEDn {train_year} alpha={alpha} seed={seed} | test: {train_year}")
        _, unlabeled_probs, unlabeled_targets = get_preds(
            data_type, net, device, 0.5, train_year,
            combine, sentence, clean, add, gemini, flip, seed
        )
        all_probs.append(unlabeled_probs[:, 0])
        all_targets.append(unlabeled_targets)

    year_dict[train_year] = (np.concatenate(all_probs), np.concatenate(all_targets))


def plot_ecdf(ax, x, label, color):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", label=label, color=color)


def plot_overlaid_hists(data_dict, bins=50, density=True, hist_alpha=0.4):
    """
    data_dict[train_year] = (probs, targets)
    Each entry is a model trained+tested on that year.
    Produces four plots: pos PDF, pos CDF, neg PDF, neg CDF.
    """
    year_colors = {2010: "blue", 2016: "purple", 2020: "red"}
    keys = list(data_dict.keys())
    cols = [year_colors[k] for k in keys]

    configs = [
        ("pos", "P(LLM | LLM)", 0, lambda p: p),
        ("neg", "P(human | human)", 1, lambda p: 1 - p),
    ]

    for suffix, xlabel, target_val, transform in configs:
        # PDF
        fig, ax = plt.subplots()
        for i, (key, (probs, targets)) in enumerate(data_dict.items()):
            probs = np.asarray(probs)
            targets = np.asarray(targets)
            mask = targets == target_val
            ax.hist(
                transform(probs[mask]),
                bins=bins,
                density=density,
                alpha=hist_alpha,
                histtype="stepfilled",
                label=str(key),
                color=cols[i],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Count")
        ax.legend(title="Train year")
        plt.tight_layout()
        plt.savefig(f"pu_temporal_{suffix}.pdf", format="pdf", bbox_inches="tight")
        plt.clf()

        # CDF
        fig, ax = plt.subplots()
        for i, (key, (probs, targets)) in enumerate(data_dict.items()):
            probs = np.asarray(probs)
            targets = np.asarray(targets)
            mask = targets == target_val
            plot_ecdf(ax, transform(probs[mask]), label=str(key), color=cols[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.legend(title="Train year")
        plt.tight_layout()
        plt.savefig(f"pu_temporal_{suffix}_cdf.pdf", format="pdf", bbox_inches="tight")
        plt.clf()


plot_overlaid_hists(year_dict)
print("Saved: pu_temporal_pos.pdf, pu_temporal_pos_cdf.pdf, pu_temporal_neg.pdf, pu_temporal_neg_cdf.pdf")
