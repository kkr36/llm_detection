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

PREDS_BASE = "/share/garg/arxiv_kaggle/predictions"

# Use all 16 CPU cores for intra-op parallelism (matrix mults inside forward pass)
torch.set_num_threads(16)
torch.set_num_interop_threads(1)


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
entrance_path = "logging_accuracy_llm_gemini"
data_type = "ArXiv_BERT"
combine = "combine" in entrance_path
sentence = True
clean = True
platt = False
gemini = "gemini" in entrance_path
add = "_add_" in entrance_path
epochs = 3
device = 'cpu'  # CPU-only; parallelism comes from torch.set_num_threads above
train_year = 2020
llms_list = ["Gemini 2.5 Pro", "Gemini 2.0 Flash-Lite", "Gemini 3 Preview", "Gemini 2.0 Flash", "Gemini 2.5 Flash"] if gemini else ["Qwen", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct", "all"][:-1]
seeds = 5

### LOGIC ###

metrics_dict = defaultdict(list)

output_csv = f"{entrance_path}_3.csv"

if os.path.exists(output_csv):
    metrics_df = pd.read_csv(output_csv)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

for train_llm in llms_list:

    train_llm_name = train_llm.replace(' ', '_')

    alphas = [0, .5]

    for train_alpha in alphas:

        eval_flip = True

        nets = []

        for n in range(seeds):

            alpha_dir = Path(f"{entrance_path}/normal_sentence/alpha_{train_alpha}/{train_llm_name}_{n}/llm_type_{train_llm_name}_{epochs}")
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
        test_year = 2020

        for test_alpha in test_alphas:
            for test_llm in llms_list:
                print(f"train: {model_name} {train_llm} {train_alpha} | test: {test_llm} {test_alpha}")

                preds_p_list, preds_u_list, u_targets_list = [], [], []

                test_llm_name = test_llm.replace(" ", "_")
                for n in range(seeds):
                    pos_probs, unlabeled_probs, unlabeled_targets = get_preds_llm(data_type, nets[n], device, test_alpha, test_year, test_llm, sentence, clean, gemini, eval_flip, n)
                    if model_name == "PN":
                        pos_probs = 1 - pos_probs
                        unlabeled_probs = 1 - unlabeled_probs
                    save_preds(
                        f"{PREDS_BASE}/heatmap/{model_name}/train_{train_llm_name}/alpha_{train_alpha}/test_{test_llm_name}/seed_{n}.npz",
                        pos_probs, unlabeled_probs, unlabeled_targets,
                    )
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
                    "clean": clean,
                    "sentence": sentence,
                    "gemini": gemini,
                    "eval_flip": eval_flip,
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

                metrics_df.to_csv(output_csv, index=False)
