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

def update_dict(metrics_dict, metric, point, lowers, uppers):
    metrics_dict[metric] = point
    for ci in uppers:
        assert(ci in lowers)
        metrics_dict[f'{metric}_l_{ci}'] = lowers[ci]
        metrics_dict[f'{metric}_u_{ci}'] = uppers[ci]

def get_metrics(preds_p, preds_u, u_targets, test_cis, n_bootstrap):

    preds_up_list, preds_un_list = [], []

    for i in range(len(preds_u)):
        preds_up = preds_u[i][u_targets[i]==0][:,0]
        preds_un = preds_u[i][u_targets[i]==1][:,0]

        preds_up_list.append(preds_up)
        preds_un_list.append(preds_un)

    metrics_dict = {}

    print('calculating metrics')

    update_dict(metrics_dict, "auc", *bootstrap_metric(auc_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "pos_prob", *bootstrap_metric(pos_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "neg_prob", *bootstrap_metric(neg_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "avg_pos_neg_prob", *bootstrap_metric(avg_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "tpr", *bootstrap_metric(tpr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "fnr", *bootstrap_metric(fnr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "tnr", *bootstrap_metric(tnr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "fpr", *bootstrap_metric(fpr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "plugin", *bootstrap_metric(plugin_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "plugin-int", *bootstrap_metric(plugin_int_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "entropy", *bootstrap_metric(binary_entropy_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "entropy_pos", *bootstrap_metric(binary_entropy_pos_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "entropy_neg", *bootstrap_metric(binary_entropy_neg_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "bbe", *bootstrap_metric_bbe(BBE_estimator, preds_p, preds_u, u_targets, n_bootstrap=n_bootstrap, cis=test_cis))

    return metrics_dict

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
epochs = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# TEDn evaluation: alpha=0.5, all years, all seeds, test_alpha=0.5
train_years = [2010, 2012, 2014, 2016, 2018, 2020]
train_alpha = 0.5
train_method = "TEDn"
test_alpha = 0.5
test_cis = [.9, .95, .99]

output_csv = f"{entrance_path}_tedn_alpha05_temporal.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

for train_year in train_years:
    nets = {}

    for seed in seeds:
        alpha_dir = Path(f"{entrance_path}/sentence_{train_year}/{train_alpha}_{seed}/ArXiv_BERT_{epochs}")
        pts = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and train_method in p.name]
        assert(len(pts) == 1), f"{train_year} {seed} bad"

        model_path = pts[0]

        net = get_model("DistilBert")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)
        nets[seed] = net

    test_years = [train_year]

    for test_year in test_years:
        print(f"train: {train_method} {train_year} {train_alpha} | test: {test_year} {test_alpha}")
        preds_p_list, preds_u_list, u_targets_list = [], [], []

        for seed in seeds:
            seednet = nets[seed]
            if platt:
                assert(not combine)
                scale_year = 2010
                u_data_loader, _, _ = get_u_data(data_type, 0.5, scale_year, combine, sentence, clean, add, gemini, flip, "out", seed)
                platt_scaler = fit_platt_scaler(
                    model=seednet,
                    calib_loader=u_data_loader,
                    device=device
                )
                seednet = PlattCalibratedClassifier(seednet, platt_scaler)
                seednet.eval()
            pos_probs, unlabeled_probs, unlabeled_targets = get_preds(data_type, seednet, device, test_alpha, test_year, combine, sentence, clean, add, gemini, flip, seed)
            preds_p_list.append(pos_probs)
            preds_u_list.append(unlabeled_probs)
            u_targets_list.append(unlabeled_targets)

        info = {
            "learning_method": train_method + (f"_platt_{scale_year}" if platt else ''),
            "data_type": data_type,
            "train_alpha": train_alpha,
            "train_year": train_year,
            "test_alpha": test_alpha,
            "test_year": test_year,
            "epochs": epochs,
            "combine": combine,
            "clean": clean,
            "sentence": sentence,
            "add": add,
            "gemini": gemini,
            "flip": flip,
            "model_path": model_path,
            "run_id": run_id
        }

        metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list, test_cis=test_cis, n_bootstrap=2500)

        row = {}
        row.update(info)
        row.update(metrics)

        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([row])],
            ignore_index=True
        )

        # save after every run (crash-safe)
        metrics_df.to_csv(output_csv, index=False)
