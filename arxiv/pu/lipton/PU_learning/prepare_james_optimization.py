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
from model_inference import get_preds_llm, get_u_data_llm
from collections import defaultdict
from model_helper import *

from prepare_metrics import *
from estimator import BBE_estimator
import torch
from platt_scaling import *
from model_inference import *

# enter: entrance file, alphas, prior csv (none == make a blank csv, path == append to the existing csv and save to new name? eg have an index 0 that keeps going up each time you pass it in)
import argparse

parser = argparse.ArgumentParser(description="Example script with entrance_path argument")

parser.add_argument(
    "--entrance_path",
    type=str,
    required=True,
    help="Path to the entrance file or directory"
)

args = parser.parse_args()

print(f"Entrance path: {args.entrance_path}")

# SWITCHES
entrance_path = args.entrance_path
data_type = "ArXiv_BERT"
# flip = False
combine = "combine" in entrance_path
sentence = True 
clean = True 
gemini = "gemini" in entrance_path
add = "_add_" in entrance_path
train_year = 2020
seeds = 5
flip = False

### LOGIC ###

output_csv = f"{entrance_path}.csv"

if not os.path.exists(os.path.expanduser(f"{data_dir}/multillm/james_v_us")):
    os.makedirs(os.path.expanduser(f"{data_dir}/multillm/james_v_us"))

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

train_alpha=0

for seed in range(seeds):
    # tokenize train
    tokenize_fn(data_type, train_year, train_alpha, combine, sentence, clean, add, gemini, flip, "train", seed, "xy")
    # do estimation on train
    estimate_train(data_type, train_year, train_alpha, combine, sentence, clean, add, gemini, flip, "train", seed, "xy")

test_alphas = [0.5]
test_cis = [.9, .95, .99]
test_year = 2020

for test_alpha in test_alphas:

    for seed in range(seeds):
        # tokenize the test set
        tokenize_fn(data_type, test_year, test_alpha, combine, sentence, clean, add, gemini, flip, "val", seed, "xy")

    # MLE with (train, test)
    alpha_hat, half_widths = MLE_james(data_type, train_year, train_alpha, test_year, test_alpha, combine, sentence, clean, add, gemini, flip, test_cis, 2500, seeds, "xy")
    # store results
    row = {
        "learning_method": "MLE",
        "data_type": data_type,
        "train_alpha": train_alpha,
        "train_year": train_year,
        "test_alpha": test_alpha,
        "test_year": test_year,
        "clean": clean,
        "sentence": sentence,
        "gemini": gemini,
        "flip": flip,
        "run_id": run_id
    }
    
    row["bbe"] = alpha_hat
    for ci in test_cis:
        row[f"bbe_l_{ci}"], row[f"bbe_u_{ci}"] = alpha_hat - half_widths[ci], alpha_hat + half_widths[ci]

    # append
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([row])],
        ignore_index=True
    )

    # save after every run (crash-safe)
    metrics_df.to_csv(output_csv, index=False)
