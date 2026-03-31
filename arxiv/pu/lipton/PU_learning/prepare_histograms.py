# for each alpha:
    # load in models, given path (both the pt file with pn and the pt file with pu)
    
    # for each model:
        # given path, parse out what experiments you want to rerun/ assemble test sets:
            # if combine, need to fix 2014-2020 (or whatever interval; most recently 2018-2020)
        # for each test set:
            # get: preds (save these in the same folder with year and alpha)
            # bbe (keep upper/lower conf bounds returned by function), 
            # avg pred pos / neg / avg(avg pos, avg neg) / avg(tpr, fpr) aka plugin (bootstrap 90% bounds?)
            # put into a new csv (train_year, train_method, train_alpha, test_alpha, test_year, **test_metrics); save/add to global df

import os
import pandas as pd
from pathlib import Path
import numpy as np
from model_inference import get_preds, get_u_data
from collections import defaultdict
from model_helper import *

from prepare_metrics import *
from estimator import BBE_estimator
import torch
from platt_scaling import *


# enter: entrance file, alphas, prior csv (none == make a blank csv, path == append to the existing csv and save to new name? eg have an index 0 that keeps going up each time you pass it in)

# SWITCHES
entrance_path = "logging_accuracy_temporal_alpha_full_sentence"
data_type = "ArXiv_BERT"
flip = False
combine = "combine" in entrance_path
sentence = True 
clean = True 
platt = False 
gemini = False
add = "_add_" in entrance_path
epochs = 3 # can toggle
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # can toggle
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][:1]

### LOGIC ###

train_years = [2010, 2020]
if combine: train_years = [2020]

output_csv = f"{entrance_path}_alpha_temporal.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

# keep 3 hists for applying (0)/2010 models to (2010,2016,2020), (0,.3,.6)/2020 models to (2020)
alpha_dict = defaultdict(lambda: {})
temp_dict = defaultdict(lambda: {})

for train_year in train_years:
    if train_year == 2020:
        alphas = [0, .3, .6]
    elif train_year == 2010:
        alphas = [0]

    for alpha in alphas:
        model_nets = {"PU": {}, "PN": {}}

        for seed in seeds:
            alpha_dir = Path(f"{entrance_path}/{'sentence' if sentence else 'abstract'}_{train_year}/{alpha}_{seed}/ArXiv_BERT_{epochs}")
            pts_pu = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "TEDn" in p.name][0]
            pts_pn = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "PN" in p.name][0]

            for model_name, model_path in [("PU", pts_pu), ("PN", pts_pn)]:
                net = get_model("DistilBert")
                state_dict = torch.load(model_path, map_location=device)
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                net.load_state_dict(state_dict)
                net.eval()
                net.to(device)
                model_nets[model_name][seed] = net

        test_years = [train_year] if train_year != 2010 else [2010, 2016, 2020]

        for model_name, nets in model_nets.items():
            for test_year in test_years:
                print(f"train: {model_name} {train_year} {alpha} | test: {test_year}")
                all_probs = []
                all_targets = []

                for seed, net in nets.items():
                    _, unlabeled_probs, unlabeled_targets_seed = get_preds(
                        data_type, net, device, 0.5, test_year,
                        combine, sentence, clean, add, gemini, flip, seed
                    )
                    all_probs.append(unlabeled_probs[:, 0])
                    all_targets.append(unlabeled_targets_seed)

                avg_probs = np.concatenate(all_probs)
                unlabeled_targets = np.concatenate(all_targets)

                if train_year == 2010:
                    temp_dict[model_name][test_year] = (avg_probs, unlabeled_targets)
                elif train_year == 2020:
                    alpha_dict[model_name][alpha] = (avg_probs, unlabeled_targets)
                    
import numpy as np
import matplotlib.pyplot as plt
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20
    }
import matplotlib
matplotlib.rc('font', **font)
from matplotlib.ticker import MaxNLocator

def plot_overlaid_hists(
    data_dict,
    model_name,
    title_prefix,
    bins=50,
    density=True,
    alpha=0.4,
):
    """
    data_dict[key] = (probs, targets)
    Produces four plots:
      - label == 1 (PDF)
      - label == 0 (PDF)
      - label == 1 (CDF)
      - label == 0 (CDF)
    """
    cols = ["blue", "purple", "red"]  # ordered, perceptual

    def plot_ecdf(ax, x, label, color):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", label=label, color=color)

    # -------- label == 1 (target == 0): PDF --------
    fig, ax = plt.subplots()
    for i, (key, (probs, targets)) in enumerate(data_dict.items()):
        probs = np.asarray(probs)
        targets = np.asarray(targets)
        mask = targets == 0

        ax.hist(
            probs[mask],
            bins=bins,
            density=density,
            alpha=alpha,
            histtype="stepfilled",
            label=str(key),
            color=cols[i],
        )

    if "year" in title_prefix.lower():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

    ax.set_xlabel("P(LLM)")
    ax.set_ylabel("Density" if density else "Count")
    ax.legend(title=title_prefix)
    plt.tight_layout()
    plt.savefig(
        f"{title_prefix.replace(' ', '_')}_{model_name}_pos.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.clf()

    # -------- label == 1 (target == 0): CDF --------
    fig, ax = plt.subplots()
    for i, (key, (probs, targets)) in enumerate(data_dict.items()):
        probs = np.asarray(probs)
        targets = np.asarray(targets)
        mask = targets == 0

        plot_ecdf(ax, probs[mask], label=str(key), color=cols[i])

    if "year" in title_prefix.lower():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

    ax.set_xlabel("P(LLM)")
    ax.set_ylabel("CDF")
    ax.legend(title=title_prefix)
    plt.tight_layout()
    plt.savefig(
        f"{title_prefix.replace(' ', '_')}_{model_name}_pos_cdf.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.clf()

    # -------- label == 0 (target == 1): PDF --------
    fig, ax = plt.subplots()
    for i, (key, (probs, targets)) in enumerate(data_dict.items()):
        probs = np.asarray(probs)
        targets = np.asarray(targets)
        mask = targets == 1

        ax.hist(
            1 - probs[mask],
            bins=bins,
            density=density,
            alpha=alpha,
            histtype="stepfilled",
            label=str(key),
            color=cols[i],
        )

    if "year" in title_prefix.lower():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

    ax.set_xlabel("P(human)")
    ax.set_ylabel("Density" if density else "Count")
    ax.legend(title=title_prefix)
    plt.tight_layout()
    plt.savefig(
        f"{title_prefix.replace(' ', '_')}_{model_name}_neg.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.clf()

    # -------- label == 0 (target == 1): CDF --------
    fig, ax = plt.subplots()
    for i, (key, (probs, targets)) in enumerate(data_dict.items()):
        probs = np.asarray(probs)
        targets = np.asarray(targets)
        mask = targets == 1

        plot_ecdf(ax, 1 - probs[mask], label=str(key), color=cols[i])

    if "year" in title_prefix.lower():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

    ax.set_xlabel("P(human)")
    ax.set_ylabel("CDF")
    ax.legend(title=title_prefix)
    plt.tight_layout()
    plt.savefig(
        f"{title_prefix.replace(' ', '_')}_{model_name}_neg_cdf.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.clf()


for model_name in temp_dict:
    plot_overlaid_hists(
        data_dict=temp_dict[model_name],
        model_name=model_name,
        title_prefix="Test year",
    )

for model_name in alpha_dict:
    plot_overlaid_hists(
        data_dict=alpha_dict[model_name],
        model_name=model_name,
        title_prefix="Alpha",
    )
