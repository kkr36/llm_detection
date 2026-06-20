import os
import pandas as pd
from pathlib import Path
import numpy as np

import torch

from model_inference import get_preds_raid, get_preds_raid_shift
from model_helper import get_model
from prepare_metrics import (
    bootstrap_metric, bootstrap_metric_bbe,
    auc_fn, pos_prob_fn, neg_prob_fn, avg_prob_fn,
    tpr_fn, fnr_fn, tnr_fn, fpr_fn,
    plugin_fn, plugin_int_fn,
    binary_entropy_fn, binary_entropy_pos_fn, binary_entropy_neg_fn,
    balanced_cross_entropy_fn,
)
from estimator import BBE_estimator
from data_helper.raid import RAID_ATTACKS

TEMPORAL_LOG_BASE = "logging_accuracy_temporal_alpha_full_sentence"
RAID_LOG_BASE = "/share/garg/arxiv_kaggle/logging_accuracy_raid"
PREDS_BASE = "/share/garg/arxiv_kaggle/predictions"

TRAIN_YEAR = 2010
TRAIN_ALPHA = 0.0
TRAIN_METHOD = 'PN'
SEEDS = [0, 1]
EPOCHS = 3

TEST_ALPHA = 0.5
TEST_CIS = [0.90, 0.95, 0.99]
N_BOOTSTRAP = 2500

OUTPUT_CSV = "logging_accuracy_temporal_raid_eval.csv"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def _alpha_dir(alpha):
    """Format alpha for the directory name: 0.0 -> '0', 0.5 -> '0.5'."""
    return str(int(alpha)) if float(alpha) == int(alpha) else str(alpha)


def _get_shift_vals(shift_col):
    """Collect all unique individual values from the raid shift directories for shift_col."""
    shift_dir = Path(RAID_LOG_BASE) / f"raid_{shift_col}"
    if not shift_dir.exists():
        return []
    vals = set()
    for sub in sorted(shift_dir.iterdir()):
        if not sub.is_dir():
            continue
        for part in sub.name.split(':'):
            vals.add(part)
    return sorted(vals)


ATTACK_TEST_VALS = ['none'] + RAID_ATTACKS
DOMAIN_VALS = _get_shift_vals('domain')
REP_PEN_VALS = _get_shift_vals('repetition_penalty')
DECODING_VALS = _get_shift_vals('decoding')


def save_preds(path, pos_probs, unlabeled_probs, unlabeled_targets):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
                        pos_probs=pos_probs,
                        unlabeled_probs=unlabeled_probs,
                        unlabeled_targets=unlabeled_targets)


def update_dict(d, metric, point, lowers, uppers):
    d[metric] = point
    for ci in uppers:
        d[f'{metric}_l_{ci}'] = lowers[ci]
        d[f'{metric}_u_{ci}'] = uppers[ci]


def get_metrics(preds_p, preds_u, u_targets, test_cis, n_bootstrap):
    preds_up_list, preds_un_list = [], []
    for i in range(len(preds_u)):
        preds_up_list.append(preds_u[i][u_targets[i] == 0][:, 0])
        preds_un_list.append(preds_u[i][u_targets[i] == 1][:, 0])

    d = {}
    for name, fn in [
        ('auc',              auc_fn),
        ('pos_prob',         pos_prob_fn),
        ('neg_prob',         neg_prob_fn),
        ('avg_pos_neg_prob', avg_prob_fn),
        ('tpr',              tpr_fn),
        ('fnr',              fnr_fn),
        ('tnr',              tnr_fn),
        ('fpr',              fpr_fn),
        ('plugin',           plugin_fn),
        ('plugin-int',       plugin_int_fn),
        ('entropy',          binary_entropy_fn),
        ('entropy_pos',      binary_entropy_pos_fn),
        ('entropy_neg',      binary_entropy_neg_fn),
        ('bce',              balanced_cross_entropy_fn),
    ]:
        try:
            update_dict(d, name, *bootstrap_metric(fn, preds_up_list, preds_un_list,
                                                    n_bootstrap=n_bootstrap, cis=test_cis))
        except Exception:
            d[name] = float('nan')
            for ci in test_cis:
                d[f'{name}_l_{ci}'] = float('nan')
                d[f'{name}_u_{ci}'] = float('nan')

    try:
        update_dict(d, 'bbe', *bootstrap_metric_bbe(BBE_estimator, preds_p, preds_u, u_targets,
                                                     n_bootstrap=n_bootstrap, cis=test_cis))
    except Exception:
        d['bbe'] = float('nan')
        for ci in test_cis:
            d[f'bbe_l_{ci}'] = float('nan')
            d[f'bbe_u_{ci}'] = float('nan')

    return d


# ── Load temporal models (once for all evaluations) ──────────────────────────
nets = {}
for seed in SEEDS:
    model_dir = Path(
        f"{TEMPORAL_LOG_BASE}/sentence_{TRAIN_YEAR}"
        f"/{_alpha_dir(TRAIN_ALPHA)}_{seed}/ArXiv_BERT_{EPOCHS}"
    )
    if not model_dir.exists():
        print(f"[skip] {model_dir} not found")
        continue
    pts = [p for p in model_dir.iterdir()
           if p.is_file() and p.name.lower().endswith('.pt') and TRAIN_METHOD in p.name]
    if len(pts) != 1:
        print(f"[skip] expected 1 .pt in {model_dir}, found {len(pts)}")
        continue
    net = get_model('DistilBert')
    state = torch.load(pts[0], map_location=device)
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    net.to(device)
    nets[seed] = net

if not nets:
    raise RuntimeError("No temporal models found — check TEMPORAL_LOG_BASE and SEEDS")

available_seeds = sorted(nets.keys())

if os.path.exists(OUTPUT_CSV):
    metrics_df = pd.read_csv(OUTPUT_CSV)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

# ── Attack evaluation ─────────────────────────────────────────────────────────
for test_attack in ATTACK_TEST_VALS:
    print(f"[attack] test_attack={test_attack} alpha={TEST_ALPHA}")

    preds_p_list, preds_u_list, u_targets_list = [], [], []
    for seed in available_seeds:
        print(f"  seed {seed}: getting preds")
        pos_probs, unlabeled_probs, unlabeled_targets = get_preds_raid(
            nets[seed], device, test_attack, TEST_ALPHA, seed
        )
        pos_probs = 1 - pos_probs
        unlabeled_probs = 1 - unlabeled_probs
        save_preds(
            f"{PREDS_BASE}/temporal_raid/attack/{test_attack}/seed_{seed}.npz",
            pos_probs, unlabeled_probs, unlabeled_targets,
        )
        preds_p_list.append(pos_probs)
        preds_u_list.append(unlabeled_probs)
        u_targets_list.append(unlabeled_targets)

    info = {
        'train_year':   TRAIN_YEAR,
        'train_method': TRAIN_METHOD,
        'train_alpha':  TRAIN_ALPHA,
        'test_col':     'attack',
        'test_val':     test_attack,
        'test_alpha':   TEST_ALPHA,
        'epochs':       EPOCHS,
        'run_id':       run_id,
    }
    metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list,
                          test_cis=TEST_CIS, n_bootstrap=N_BOOTSTRAP)
    metrics_df = pd.concat([metrics_df, pd.DataFrame([{**info, **metrics}])], ignore_index=True)
    metrics_df.to_csv(OUTPUT_CSV, index=False)
    run_id += 1

# ── Shift evaluation (domain, repetition_penalty, decoding) ──────────────────
for shift_col, test_vals in [
    # ('domain',             DOMAIN_VALS),
    # ('repetition_penalty', REP_PEN_VALS),
    ('decoding',           DECODING_VALS[-1:]),
]:
    for test_val in test_vals:
        print(f"[shift] {shift_col}={test_val} alpha={TEST_ALPHA}")

        preds_p_list, preds_u_list, u_targets_list = [], [], []
        for seed in available_seeds:
            print(f"  seed {seed}: getting preds")
            pos_probs, unlabeled_probs, unlabeled_targets = get_preds_raid_shift(
                nets[seed], device, shift_col, test_val, TEST_ALPHA, seed
            )
            pos_probs = 1 - pos_probs
            unlabeled_probs = 1 - unlabeled_probs
            save_preds(
                f"{PREDS_BASE}/temporal_raid/{shift_col}/{test_val}/seed_{seed}.npz",
                pos_probs, unlabeled_probs, unlabeled_targets,
            )
            preds_p_list.append(pos_probs)
            preds_u_list.append(unlabeled_probs)
            u_targets_list.append(unlabeled_targets)

        info = {
            'train_year':   TRAIN_YEAR,
            'train_method': TRAIN_METHOD,
            'train_alpha':  TRAIN_ALPHA,
            'test_col':     shift_col,
            'test_val':     test_val,
            'test_alpha':   TEST_ALPHA,
            'epochs':       EPOCHS,
            'run_id':       run_id,
        }
        metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list,
                              test_cis=TEST_CIS, n_bootstrap=N_BOOTSTRAP)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{**info, **metrics}])], ignore_index=True)
        metrics_df.to_csv(OUTPUT_CSV, index=False)
        run_id += 1
