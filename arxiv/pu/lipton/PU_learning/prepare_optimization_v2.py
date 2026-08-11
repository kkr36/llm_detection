"""
Iteration-parametrized re-evaluation of the two adversarial-mirror rows from
logging_accuracy_xz.csv, on the v2 (regenerated) arXiv parquet.

For a given rewrite iteration k (env REWRITE_ITER, default 1; also accepts argv[1]):
  TEDn row: train_llm = "x" + "z"*(k+1), eval on rewrite_Z_{k}_PU   (alpha=0.25, flip=True)
  PN   row: train_llm = "x" + "z"*k,     eval on rewrite_Z_{k}_PN   (alpha=0,    flip=False)

  k=1 -> (TEDn xzz -> rewrite_Z_1_PU), (PN xz  -> rewrite_Z_1_PN)
  k=2 -> (TEDn xzzz-> rewrite_Z_2_PU), (PN xzz -> rewrite_Z_2_PN)   [needs Z_2 regenerated + models trained]

The eval parquet is selected via the XY_EVAL_PARQUET env var (consumed inside model_inference's
get_preds_xy / get_u_data_xy). Point it at the v2 parquet before running.

Differences vs prepare_optimization.py (everything else — get_metrics, model loading, bootstrap
CIs, n_bootstrap — is identical so numbers stay comparable):
  * configs are built from the iteration k (not a hardcoded 9-entry list)
  * output_csv = logging_accuracy_xz_v2.csv (appended; original csv untouched)
  * prediction cache namespaced under predictions/optimization_v2/ (old optimization/ preserved)
  * guardrails fail loudly if the eval column or the trained models are missing for this k
"""
import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from model_inference import get_preds_xy, _xy_eval_parquet
from collections import defaultdict
from model_helper import *

from prepare_metrics import *
from estimator import BBE_estimator
import torch


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
    update_dict(metrics_dict, "bce", *bootstrap_metric(balanced_cross_entropy_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
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

# Rewrite iteration k (which rewrite_Z_{k}_{method} adversarial column to evaluate on).
ITER = int(os.environ.get("REWRITE_ITER", sys.argv[1] if len(sys.argv) > 1 else 1))

# Per-iteration configs (see module docstring / logging_accuracy_xz.csv rows 2 & 6 for k=1):
#   TEDn train_llm has (k+1) z's; PN train_llm has k z's; both eval on rewrite_Z_{k}_{method}.
configs = [
    {"train_method": "TEDn", "train_alpha": 0.25, "flip": True,  "llm": "x" + "z" * (ITER + 1), "eval_cols": [f"rewrite_Z_{ITER}_PU"]},
    {"train_method": "PN",   "train_alpha": 0,    "flip": False, "llm": "x" + "z" * ITER,       "eval_cols": [f"rewrite_Z_{ITER}_PN"]},
]

### LOGIC ###

eval_parquet = _xy_eval_parquet()
print(f"[prepare_optimization_v2] iteration k={ITER} | eval parquet = {eval_parquet}")
print(f"[prepare_optimization_v2] configs = {configs}")

# Guardrail 1: the eval columns for this iteration must exist in the eval parquet.
available_cols = set(pq.read_schema(eval_parquet).names)
for cfg in configs:
    for col in cfg["eval_cols"]:
        if col not in available_cols:
            m = "PU" if col.endswith("_PU") else "PN"
            raise SystemExit(
                f"Eval column '{col}' not found in {eval_parquet}.\n"
                f"Regenerate it first, e.g.:\n"
                f"  sbatch arxiv/inference_set_rewrite/iterative_prompt_rewrite_scale/run_xmirror_z1.sbatch {m}\n"
                f"  (or: python rewrite_x_mirror_z1.py {m} {ITER})\n"
                f"then rerun this eval for iteration k={ITER}."
            )

output_csv = "logging_accuracy_xz_v2.csv"

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
        pts = ([p for p in alpha_dir.iterdir()
                if p.is_file() and p.name.lower().endswith(".pt") and train_method in p.name]
               if alpha_dir.is_dir() else [])
        # Guardrail 2: the trained models for this (method, llm) must exist for iteration k.
        assert len(pts) == 1, (
            f"Expected 1 {train_method} .pt for seed {seed} in {alpha_dir}, found {len(pts)}.\n"
            f"For iteration k={ITER} you must first train TEDn llm='x'+'z'*(k+1) and "
            f"PN llm='x'+'z'*k on the v2 parquet (run_xy_v2_array.sbatch pattern)."
        )
        model_path = pts[0]

        net = get_model("DistilBert")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)
        nets[seed] = net

    eval_cols = ["rewrite_X", "rewrite_Y", "rewrite_Z"] if "eval_cols" not in cfg.keys() else cfg["eval_cols"]
    print(cfg, eval_cols)

    for eval_llm_col in eval_cols:
        print(f"Evaluating {train_method} llm={llm} (flip={flip}, train_alpha={train_alpha}) "
              f"on eval_col={eval_llm_col} across {len(nets)} seeds")

        preds_p_list, preds_u_list, u_targets_list = [], [], []
        for seed, net in nets.items():
            pos_probs, unlabeled_probs, unlabeled_targets = get_preds_xy(
                net, device, test_alpha, flip, seed, sentence, clean, eval_llm_col)
            save_preds(
                f"{PREDS_BASE}/optimization_v2/{train_method}/{llm}/{eval_llm_col}/seed_{seed}.npz",
                pos_probs, unlabeled_probs, unlabeled_targets,
            )
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
            "iteration": ITER,
            "eval_parquet": eval_parquet,
            "run_id": run_id,
        }

        metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list, test_cis=test_cis, n_bootstrap=2500)

        row = {}
        row.update(info)
        row.update(metrics)

        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        metrics_df.to_csv(output_csv, index=False)

        run_id += 1
