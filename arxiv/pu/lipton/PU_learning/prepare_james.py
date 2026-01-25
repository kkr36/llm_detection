import os
import pandas as pd
from pathlib import Path
import numpy as np
from model_inference import get_preds
from collections import defaultdict
from model_helper import *

from prepare_metrics import *
from james_methods import *

train_years = [2010, 2012, 2014, 2016, 2018, 2020]

entrance_path = "logging_accuracy_temporal_alpha_full_sentence_combine"

data_type = "ArXiv_BERT"

flip = False

combine = "combine" in entrance_path

sentence = True # can toggle

clean = True # can toggle

gemini = False

add = False

output_csv = f"{entrance_path}_alpha_temporal.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

# for each train_year:
for train_year in train_years:
    if train_year == 2020:
        alphas = [0, .15, .3, .45, .6]
    elif train_year == 2018: 
        alphas = [0, 0.44999999999999996] if train_year == 2018 else [0, .45]
    elif train_year == 2016:
        alphas = [0, .3]
    elif train_year == 2014:
        alphas = [0, .15]
    elif train_year == 2010 or train_year == 2010:
        alphas = [0]

    # for each train_alpha:
    for alpha in alphas:
        # tokenize train
        tokenize_fn(data_type, train_year, alpha, combine, sentence, clean, add, gemini, flip)
        # do estimation on train
        estimate_train(data_type, train_year, alpha, combine, sentence, clean, add, gemini, flip)
        # get test_years, test_alpha
        test_alphas = [0.5]
        # if alpha not in test_alphas: test_alphas.append(alpha)
        test_years = [train_year] if train_year != 2010 else train_years

        # for each test_alpha:
        for test_alpha in test_alphas:
            # for each test_year:
            for test_year in test_years:

                # tokenize the test set
                tokenize_fn(data_type, test_year, test_alpha, combine, sentence, clean, add, gemini, flip)
                # MLE with (train, test)
                alpha_hat, ci = MLE(data_type, train_year, alpha, test_year, test_alpha, combine, sentence, clean, add, gemini, flip)
                # store results
                row = {}
                row["bbe"], row["bbe_l"], row["bbe_u"] = alpha_hat, alpha_hat - ci, alpha_hat + ci

                # append
                metrics_df = pd.concat(
                    [metrics_df, pd.DataFrame([row])],
                    ignore_index=True
                )

                # save after every run (crash-safe)
                metrics_df.to_csv(output_csv, index=False)