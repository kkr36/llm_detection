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

# prior_csv = None # if something, will need to load
# if prior_csv:
#     prior_df = pd.read_csv(prior_csv)
# exit_path = entrance_path if not prior_csv else f"{prior_csv.split('.csv')[0]}1.csv"

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

    # import pdb; pdb.set_trace()

    metrics_dict = {}

    # ============================================================================
    # Compute all metrics
    # ============================================================================

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
    # import pdb; pdb.set_trace()
    update_dict(metrics_dict, "bbe", *bootstrap_metric_bbe(BBE_estimator, preds_p, preds_u, u_targets, n_bootstrap=n_bootstrap, cis=test_cis))
    
    return metrics_dict

# enter: entrance file, alphas, prior csv (none == make a blank csv, path == append to the existing csv and save to new name? eg have an index 0 that keeps going up each time you pass it in)

# SWITCHES
entrance_path = "logging_accuracy_llm"
data_type = "ArXiv_BERT"
# flip = False
combine = "combine" in entrance_path
sentence = True 
clean = True 
platt = False 
gemini = "gemini" in entrance_path
add = "_add_" in entrance_path
epochs = 3 # can toggle
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # can toggle
train_year = 2020
llms_list = ["Gemini 2.5 Pro", "Gemini 2.0 Flash-Lite", "Gemini 3 Preview", "Gemini 2.0 Flash", "Gemini 2.5 Flash"] if gemini else ["Gemini 2.5 Flash", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct", "all"][:-1]
n_models = 5

### LOGIC ###

metrics_dict = defaultdict(list)

output_csv = f"{entrance_path}.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

for train_llm in llms_list:
    train_llm_name = train_llm.replace(' ', '_')

    alphas = [0, .5]

    for train_alpha in alphas:

        flip = train_alpha==0.5

        nets = []

        for n in range(n_models):

            alpha_dir = Path(f"{entrance_path}/normal_sentence/alpha_{train_alpha}/{train_llm_name}_{n}/llm_type_{train_llm_name}_{epochs}")
            # pts_pu = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "TEDn" in p.name][0]
            # pts_pn = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "PN" in p.name][0]
            pts = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt")]
            assert(len(pts) == 1)

            model_name = "PU" if "TEDn" in pts[0].name else "PN"
            model_path = pts[0]
            assert(model_name==("PU" if train_alpha==0.5 else "PN"))

            net = get_model("DistilBert")
            state_dict = torch.load(model_path, map_location=device)
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            net.load_state_dict(state_dict)
            net.eval()
            net.to(device)
            nets.append(net)

        test_alphas = [0.5]
        test_cis = [.9, .95, .99]
        # if alpha not in test_alphas: test_alphas.append(alpha)
        test_year = 2020

        # import pdb; pdb.set_trace()

        for test_alpha in test_alphas:
            for test_llm in llms_list:
                print(f"train: {model_name} {train_llm} {train_alpha} | test: {test_llm} {test_alpha}")


                preds_p_list, preds_u_list, u_targets_list = [], [], []

                for n in range(n_models):
                    pos_probs, unlabeled_probs, unlabeled_targets = get_preds_llm(data_type, nets[n], device, test_alpha, test_year, test_llm, sentence, clean, gemini, flip)
                    preds_p_list.append(pos_probs)
                    preds_u_list.append(unlabeled_probs)
                    u_targets_list.append(unlabeled_targets)

                info = {
                    "learning_method": model_name,
                    "data_type": data_type,
                    "train_alpha": train_alpha,
                    "train_year": train_year,
                    "train_llm": train_llm,
                    "test_alpha": test_alpha,
                    "test_year": test_year,
                    "test_llm": test_llm,
                    "epochs": epochs,
                    # "combine": combine,
                    "clean": clean,
                    "sentence": sentence,
                    # "add": add,
                    "gemini": gemini,
                    "flip": flip,
                    "model_path": model_path,
                    "run_id": run_id
                }

                metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list, test_cis=test_cis, n_bootstrap=2500)

                # one row per experiment
                row = {}
                row.update(info)
                row.update(metrics)

                # append
                metrics_df = pd.concat(
                    [metrics_df, pd.DataFrame([row])],
                    ignore_index=True
                )

                # save after every run (crash-safe)
                metrics_df.to_csv(output_csv, index=False)
