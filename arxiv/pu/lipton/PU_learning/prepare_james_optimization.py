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

train_alpha = 0
train_llms = ["X", "Y", "all"]
test_llms  = ["X", "Y", "all"]

# Phase 1: tokenize and estimate all training sets
n_phase1 = len(train_llms) * seeds
print(f"\n=== Phase 1: tokenize+estimate training sets ({n_phase1} jobs) ===")
for train_llm in train_llms:
    for seed in range(seeds):
        print(f"  [Phase 1] train_llm={train_llm}  seed={seed}  tokenizing...")
        tokenize_fn(data_type, train_year, train_alpha, combine, sentence, clean, add, gemini, flip, "pn_train", seed, train_llm)
        print(f"  [Phase 1] train_llm={train_llm}  seed={seed}  estimating...")
        estimate_train(data_type, train_year, train_alpha, combine, sentence, clean, add, gemini, flip, "pn_train", seed, train_llm)
print("=== Phase 1 done ===\n")

test_alphas = [0.5]
test_cis = [.9, .95, .99]
test_year = 2020

# Phase 2: tokenize all validation sets (depends only on test_llm, not train_llm)
n_phase2 = len(test_alphas) * len(test_llms) * seeds
print(f"=== Phase 2: tokenize validation sets ({n_phase2} jobs) ===")
for test_alpha in test_alphas:
    for test_llm in test_llms:
        for seed in range(seeds):
            print(f"  [Phase 2] test_alpha={test_alpha}  test_llm={test_llm}  seed={seed}  tokenizing...")
            tokenize_fn(data_type, test_year, test_alpha, combine, sentence, clean, add, gemini, flip, "cal", seed, test_llm)
print("=== Phase 2 done ===\n")

# Phase 3: run MLE for all (train_llm, test_llm) combinations
n_phase3 = len(test_alphas) * len(train_llms) * len(test_llms)
print(f"=== Phase 3: MLE estimation ({n_phase3} combinations) ===")
for test_alpha in test_alphas:
    for train_llm in train_llms:
        for test_llm in test_llms:
            print(f"  [Phase 3] train_llm={train_llm}  test_llm={test_llm}  test_alpha={test_alpha}  running MLE...")
            alpha_hat, half_widths = MLE_james(data_type, train_year, train_alpha, test_year, test_alpha, combine, sentence, clean, add, gemini, flip, test_cis, 2500, seeds, (train_llm, test_llm))
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
                "train_llm": train_llm,
                "test_llm": test_llm,
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

            print(f"    => alpha_hat={alpha_hat:.4f}  saved to {output_csv}")
            # save after every run (crash-safe)
            metrics_df.to_csv(output_csv, index=False)
print("=== Phase 3 done ===\n")
