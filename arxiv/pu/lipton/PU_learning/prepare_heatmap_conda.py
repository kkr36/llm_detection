"""Evaluate trained ConDA models and write a metrics CSV (analogue of
prepare_heatmap_pnu.py). For each (LLM1, LLM2) pair, the ConDA model trained with
source=LLM1 / target=LLM2 is evaluated on LLM2 (the target). Reuses the shared
metrics/bootstrap stack and get_preds_llm unchanged.

ConDA is trained human=positive (like PNU, --flip), so it emits P(human) directly
-> no probability inversion at eval.
"""
import os
import re
import pandas as pd
from pathlib import Path
import numpy as np
from model_inference import get_preds_llm
from collections import defaultdict

from prepare_metrics import *
from estimator import BBE_estimator
import torch

from models.conda import ConDADistilBert

CONDA_BASE = "/share/garg/arxiv_kaggle/ConDA_llm"
PREDS_BASE = "/share/garg/arxiv_kaggle/predictions"


def save_preds(path, pos_probs, unlabeled_probs, unlabeled_targets):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, pos_probs=pos_probs, unlabeled_probs=unlabeled_probs,
                        unlabeled_targets=unlabeled_targets)


def update_dict(metrics_dict, metric, point, lowers, uppers):
    metrics_dict[metric] = point
    for ci in uppers:
        assert ci in lowers
        metrics_dict[f'{metric}_l_{ci}'] = lowers[ci]
        metrics_dict[f'{metric}_u_{ci}'] = uppers[ci]


def get_metrics(preds_p, preds_u, u_targets, test_cis, n_bootstrap):
    preds_up_list, preds_un_list = [], []
    for i in range(len(preds_u)):
        preds_up = preds_u[i][u_targets[i] == 0][:, 0]
        preds_un = preds_u[i][u_targets[i] == 1][:, 0]
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
data_type = "ArXiv_BERT"
sentence = True
clean = True
gemini = "gemini" in CONDA_BASE.lower()
epochs = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_year = 2020
test_year = 2020
test_alpha = 0.5
train_alpha = 0.5
test_cis = [.9, .95, .99]
eval_flip = True  # human=positive for eval (ConDA trained human-positive, no inversion)

output_csv = "logging_accuracy_llm_conda.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)


def load_conda_net(model_path):
    net = ConDADistilBert(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    return net


# Discover all (llm1, llm2) pairs from directory structure
# Dirs are named: llm_type_{llm1}|{llm2}_{seed}
pair_pattern = re.compile(r'^llm_type_(.+)\|(.+)_(\d+)$')
pairs_seeds = defaultdict(list)  # (llm1_name, llm2_name) -> [seed, ...]

for dirname in sorted(os.listdir(CONDA_BASE)):
    m = pair_pattern.match(dirname)
    if m:
        llm1_name, llm2_name, seed = m.group(1), m.group(2), int(m.group(3))
        pairs_seeds[(llm1_name, llm2_name)].append(seed)

cfgs = sorted(pairs_seeds.items())

for (llm1_name, llm2_name), seeds in cfgs:
    seeds = sorted(seeds)

    nets = []
    model_paths = []
    for n in seeds:
        outer_dir = Path(f"{CONDA_BASE}/llm_type_{llm1_name}|{llm2_name}_{n}")
        inner_dir = outer_dir / f"llm_type_{llm1_name}|{llm2_name}_{epochs}"
        pts = [p for p in inner_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt")]
        assert len(pts) == 1, f"Expected 1 .pt file in {inner_dir}, found {len(pts)}"
        model_paths.append(pts[0])
        nets.append(load_conda_net(pts[0]))

    train_llm = f"{llm1_name}|{llm2_name}"

    # Evaluate on the target LLM2 (the goal: ConDA trained {LLM1, LLM2}, tested on LLM2).
    test_llm = llm2_name.replace("_", " ")
    test_llm_name = llm2_name

    print(f"train: ConDA {train_llm} alpha={train_alpha} | test: {test_llm} alpha={test_alpha}")

    preds_p_list, preds_u_list, u_targets_list = [], [], []
    for i, n in enumerate(seeds):
        pos_probs, unlabeled_probs, unlabeled_targets = get_preds_llm(
            data_type, nets[i], device, test_alpha, test_year, test_llm,
            sentence, clean, gemini, eval_flip, n
        )
        save_preds(
            f"{PREDS_BASE}/heatmap/ConDA/train_{llm1_name}|{llm2_name}/alpha_{train_alpha}/test_{test_llm_name}/seed_{n}.npz",
            pos_probs, unlabeled_probs, unlabeled_targets,
        )
        preds_p_list.append(pos_probs)
        preds_u_list.append(unlabeled_probs)
        u_targets_list.append(unlabeled_targets)

    info = {
        "learning_method": "ConDA",
        "data_type": data_type,
        "train_alpha": train_alpha,
        "train_year": train_year,
        "train_llm": train_llm,
        "test_alpha": test_alpha,
        "test_year": test_year,
        "test_llm": test_llm,
        "epochs": epochs,
        "clean": clean,
        "sentence": sentence,
        "gemini": gemini,
        "eval_flip": eval_flip,
        "model_path": str(model_paths[-1]),
        "run_id": run_id,
    }

    metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list, test_cis=test_cis, n_bootstrap=2500)

    row = {}
    row.update(info)
    row.update(metrics)

    metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
    metrics_df.to_csv(output_csv, index=False)
    run_id += 1
