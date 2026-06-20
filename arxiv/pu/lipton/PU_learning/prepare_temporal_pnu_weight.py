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

PREDS_BASE = "/share/garg/arxiv_kaggle/predictions"

def save_preds(path, pos_probs, unlabeled_probs, unlabeled_targets):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, pos_probs=pos_probs, unlabeled_probs=unlabeled_probs, unlabeled_targets=unlabeled_targets)

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
    update_dict(metrics_dict, "bce", *bootstrap_metric(balanced_cross_entropy_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "bbe", *bootstrap_metric_bbe(BBE_estimator, preds_p, preds_u, u_targets, n_bootstrap=n_bootstrap, cis=test_cis))

    return metrics_dict

# SWITCHES
entrance_path = "logging_accuracy_temporal"
data_type = "ArXiv_BERT"
flip = False
combine = "combine" in entrance_path
sentence = True
clean = True
gemini = False
add = "_add_" in entrance_path
epochs = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
seeds = [0, 1]

### LOGIC ###

metrics_dict = defaultdict(list)

train_years = [2016, 2018, 2020]

# Each config is a dict with keys matching the directory name components
weight_configs = [
    {"lp": 0.5, "ln": 0.0, "lup": 0.25, "lun": 0.25},
    # {"lp": 0.5, "ln": 0.5, "lup": 0.0,  "lun": 0.0},
]

def weight_dir_str(cfg):
    return f"lp{cfg['lp']:.4f}_ln{cfg['ln']:.4f}_lup{cfg['lup']:.4f}_lun{cfg['lun']:.4f}"

output_csv = "logging_accuracy_temporal_pnu_weight.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

for train_year in train_years:
    for weight_cfg in weight_configs:
        w_str = weight_dir_str(weight_cfg)
        method_name = f"PNU_weight_{w_str}"

        alpha = 0.0  # PNU_weight models use alpha=0.0 like standard PNU
        test_alpha = 0.5
        test_cis = [.9, .95, .99]
        test_years = [train_year]

        nets = {}

        for seed in seeds:
            alpha_dir = Path(f"{entrance_path}/sentence_{train_year}/PNU_weight_fix_{w_str}_{seed}/ArXiv_BERT_{epochs}")
            pts = [p for p in alpha_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "PNU" in p.name]
            assert(len(pts) == 1), f"{train_year} {w_str} {seed} bad"

            model_path = pts[0]

            net = get_model("DistilBert")
            state_dict = torch.load(model_path, map_location=device)
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            net.load_state_dict(state_dict)
            net.eval()
            net.to(device)
            nets[seed] = net

        for test_year in test_years:
            print(f"train: {method_name} {train_year} | test: {test_year} {test_alpha}")
            preds_p_list, preds_u_list, u_targets_list = [], [], []

            for seed in seeds:
                seednet = nets[seed]
                pos_probs, unlabeled_probs, unlabeled_targets = get_preds(data_type, seednet, device, test_alpha, test_year, combine, sentence, clean, add, gemini, flip, seed)
                save_preds(
                    f"{PREDS_BASE}/temporal/{method_name}/train_{train_year}/test_{test_year}/seed_{seed}.npz",
                    pos_probs, unlabeled_probs, unlabeled_targets,
                )
                preds_p_list.append(pos_probs)
                preds_u_list.append(unlabeled_probs)
                u_targets_list.append(unlabeled_targets)

            info = {
                "learning_method": method_name,
                "data_type": data_type,
                "train_alpha": alpha,
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
                "model_path": str(model_path),
                "run_id": run_id,
                **{f"weight_{k}": v for k, v in weight_cfg.items()},
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
