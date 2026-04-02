import os
import pandas as pd
from pathlib import Path
import numpy as np
from model_inference import get_preds_xy
from collections import defaultdict
from model_helper import *

from prepare_metrics import *
from estimator import BBE_estimator
import torch


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
    update_dict(metrics_dict, "pos_prob", *bootstrap_metric(pos_prob_fn, preds_un_list, preds_up_list, n_bootstrap=n_bootstrap, cis=test_cis))
    update_dict(metrics_dict, "neg_prob", *bootstrap_metric(neg_prob_fn, preds_un_list, preds_up_list, n_bootstrap=n_bootstrap, cis=test_cis))
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
entrance_path = "logging_accuracy_xy"
sentence = True
clean = True
epochs = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
seeds = [0, 1, 2, 3, 4]
test_alpha = 0.5
test_cis = [.9, .95, .99]
# Each entry is one (train_method, llm) combination to evaluate.
# For PN: alpha=0, flip=False (AI=positive)
# For TEDn: alpha=0.25, flip=True (human=positive)
configs = [
    {"train_method": "PN",   "train_alpha": 0,    "flip": False, "llm": "X"},
    {"train_method": "PN",   "train_alpha": 0,    "flip": False, "llm": "Y"},
    {"train_method": "TEDn", "train_alpha": 0.25, "flip": True,  "llm": "Y"},
    {"train_method": "TEDn", "train_alpha": 0.25, "flip": True,  "llm": "all"},
    {"train_method": "TEDn", "train_alpha": 0.25, "flip": True,  "llm": "X"},
    {"train_method": "PN",   "train_alpha": 0,    "flip": False, "llm": "Z"},
    {"train_method": "TEDn", "train_alpha": 0.25, "flip": True,  "llm": "Z"},
][-2:]

### LOGIC ###

output_csv = f"{entrance_path}.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

for cfg in configs:
    train_method = cfg["train_method"]
    train_alpha = cfg["train_alpha"]
    flip = cfg["flip"]
    llm = cfg["llm"]

    nets = {}
    model_path = None
    for seed in seeds:
        alpha_dir = Path(f"{entrance_path}/normal_sentence/alpha_{train_alpha}/{seed}/{llm}/xy_{epochs}")
        pts = [p for p in alpha_dir.iterdir()
               if p.is_file() and p.name.lower().endswith(".pt") and train_method in p.name]
        assert len(pts) == 1, f"Expected 1 .pt for {train_method} seed {seed}, found {len(pts)} in {alpha_dir}"
        model_path = pts[0]

        net = get_model("DistilBert")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)
        nets[seed] = net

    for eval_llm_col in ["rewrite_X", "rewrite_Y"]:
        print(f"Evaluating {train_method} llm={llm} (flip={flip}, train_alpha={train_alpha}) "
              f"on eval_col={eval_llm_col} across {len(nets)} seeds")

        preds_p_list, preds_u_list, u_targets_list = [], [], []
        for seed, net in nets.items():
            pos_probs, unlabeled_probs, unlabeled_targets = get_preds_xy(
                net, device, test_alpha, flip, seed, sentence, clean, eval_llm_col)
            preds_p_list.append(pos_probs)
            preds_u_list.append(unlabeled_probs)
            u_targets_list.append(unlabeled_targets)

        info = {
            "learning_method": train_method,
            "data_type": "xy",
            "train_llm": llm,
            "eval_llm": eval_llm_col,
            "train_alpha": train_alpha,
            "test_alpha": test_alpha,
            "flip": flip,
            "clean": clean,
            "sentence": sentence,
            "epochs": epochs,
            "model_dir": str(alpha_dir),
            "run_id": run_id,
        }

        metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list, test_cis=test_cis, n_bootstrap=2500)

        row = {}
        row.update(info)
        row.update(metrics)

        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        metrics_df.to_csv(output_csv, index=False)

        run_id += 1
