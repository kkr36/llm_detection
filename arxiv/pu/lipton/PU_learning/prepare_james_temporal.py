import os
import pandas as pd
from pathlib import Path
import numpy as np
from model_inference import *
from collections import defaultdict
from model_helper import *

from prepare_metrics import *

train_years = [2010, 2012, 2014, 2016, 2018, 2020]

entrance_path = "logging_accuracy_temporal"

data_type = "ArXiv_BERT"

flip = False

combine = "combine" in entrance_path

sentence = True # can toggle

clean = True # can toggle

gemini = False

add = "add" in entrance_path

output_csv = f"{entrance_path}_temporal.csv"

seeds = 5

if not os.path.exists(os.path.expanduser(f"{data_dir}/multillm/james_v_us")):
    os.makedirs(os.path.expanduser(f"{data_dir}/multillm/james_v_us"))

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

for train_year in train_years:
    alphas = [0]

    for alpha in alphas:

        for seed in range(seeds):
            # tokenize train
            tokenize_fn(data_type, train_year, alpha, combine, sentence, clean, add, gemini, flip, "train", seed, None)
            # do estimation on train
            estimate_train(data_type, train_year, alpha, combine, sentence, clean, add, gemini, flip, "train", seed, None)

        # get test_years, test_alpha
        test_alphas = [0.5]
        test_cis = [.9, .95, .99]
        test_years = [train_year] if train_year != 2010 else train_years

        for test_alpha in test_alphas:
            for test_year in test_years:

                print(f"{train_year} {alpha} | {test_year} {test_alpha}")

                for seed in range(seeds):
                    # tokenize the test set
                    tokenize_fn(data_type, test_year, test_alpha, combine, sentence, clean, add, gemini, flip, "val", seed, None)

                # MLE with (train, test)
                alpha_hat, half_widths = MLE_james(data_type, train_year, alpha, test_year, test_alpha, combine, sentence, clean, add, gemini, flip, test_cis, 2500, seeds, None)
                # store results
                row = {
                        "learning_method": "MLE",
                        "data_type": data_type,
                        "train_alpha": alpha,
                        "train_year": train_year,
                        "test_alpha": test_alpha,
                        "test_year": test_year,
                        "combine": combine,
                        "clean": clean,
                        "sentence": sentence,
                        "add": add,
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
