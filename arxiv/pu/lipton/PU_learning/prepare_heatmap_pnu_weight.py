import os
import re
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

PNU_WEIGHT_BASE = "/share/garg/arxiv_kaggle/PNU_llm/PNU_weight"
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
data_type = "ArXiv_BERT"
sentence = True
clean = True
gemini = "gemini" in PNU_WEIGHT_BASE.lower()
epochs = 3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_year = 2020
test_year = 2020
test_alpha = 0.5
train_alpha = 0.5
test_cis = [.9, .95, .99]
eval_flip = True  # always human=positive for eval

output_csv = "logging_accuracy_llm_pnu_weight.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

# Discover all (llm1, llm2, tag) combos from directory structure
# Dirs are named: llm_type_{llm1}|{llm2}_{tag}_{seed}
# where tag has the form lp{float}_ln{float}_lup{float}_lun{float}
pair_weight_pattern = re.compile(r'^llm_type_(.+)\|(.+?)_(lp[\d.]+_ln[\d.]+_lup[\d.]+_lun[\d.]+)_(\d+)$')
configs_seeds = defaultdict(list)  # (llm1, llm2, tag) -> [seed, ...]

for dirname in sorted(os.listdir(PNU_WEIGHT_BASE)):
    m = pair_weight_pattern.match(dirname)
    if m:
        llm1_name, llm2_name, tag, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        configs_seeds[(llm1_name, llm2_name, tag)].append(seed)

def config_is_complete(llm1_name, llm2_name, tag, seeds):
    for n in seeds:
        inner_dir = Path(f"{PNU_WEIGHT_BASE}/llm_type_{llm1_name}|{llm2_name}_{tag}_{n}") / f"llm_type_{llm1_name}|{llm2_name}_{epochs}"
        if not inner_dir.exists():
            return False
        pts = [p for p in inner_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt")]
        if len(pts) != 1:
            return False
    return True

cfgs = sorted(
    (key, seeds) for key, seeds in configs_seeds.items()
    if config_is_complete(*key, sorted(seeds))
)
for (llm1_name, llm2_name, tag), seeds in cfgs:
    seeds = sorted(seeds)

    nets = []
    model_paths = []

    for n in seeds:
        outer_dir = Path(f"{PNU_WEIGHT_BASE}/llm_type_{llm1_name}|{llm2_name}_{tag}_{n}")
        inner_dir = outer_dir / f"llm_type_{llm1_name}|{llm2_name}_{epochs}"
        pts = [p for p in inner_dir.iterdir() if p.is_file() and p.name.lower().endswith(".pt")]
        assert len(pts) == 1, f"Expected 1 .pt file in {inner_dir}, found {len(pts)}"

        model_path = pts[0]
        net = get_model("DistilBert")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)
        nets.append(net)
        model_paths.append(model_path)

    train_llm = f"{llm1_name}|{llm2_name}"

    for eval_llm in [llm1_name, llm2_name]:
        # evaluate on each LLM used during training
        test_llm = eval_llm.replace("_", " ")
        test_llm_name = eval_llm

        print(f"train: PNU_weight {train_llm} tag={tag} alpha={train_alpha} | test: {test_llm} alpha={test_alpha}")

        preds_p_list, preds_u_list, u_targets_list = [], [], []

        for i, n in enumerate(seeds):
            pos_probs, unlabeled_probs, unlabeled_targets = get_preds_llm(
                data_type, nets[i], device, test_alpha, test_year, test_llm,
                sentence, clean, gemini, eval_flip, n
            )
            # PNU_weight trained with --flip (LLM=positive), outputs P(LLM) directly — no flip needed
            save_preds(
                f"{PREDS_BASE}/heatmap/PNU_weight/train_{llm1_name}|{llm2_name}/{tag}/alpha_{train_alpha}/test_{test_llm_name}/seed_{n}.npz",
                pos_probs, unlabeled_probs, unlabeled_targets,
            )
            preds_p_list.append(pos_probs)
            preds_u_list.append(unlabeled_probs)
            u_targets_list.append(unlabeled_targets)

        info = {
            "learning_method": "PNU_weight",
            "data_type": data_type,
            "train_alpha": train_alpha,
            "train_year": train_year,
            "train_llm": train_llm,
            "tag": tag,
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

        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([row])],
            ignore_index=True
        )

        metrics_df.to_csv(output_csv, index=False)
        run_id += 1
