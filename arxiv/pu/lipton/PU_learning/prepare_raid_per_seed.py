import os
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

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

RAID_LOG_BASE = "/share/garg/arxiv_kaggle/logging_accuracy_raid"
PREDS_BASE = "/share/garg/arxiv_kaggle/predictions"

SEEDS = [0, 1, 2, 3, 4][2:]
EPOCHS = 3

EVAL_GROUPS = [
    # ('none', 'PN', 0.0, ['all', 'article_deletion', 'homoglyph', 'none', 'paraphrase', 'whitespace', 'alternative_spelling', 'zero_width_space', 'insert_paragraphs', 'synonym', 'perplexity_misspelling', 'number', 'upper_lower']),
    ('none', 'PN', 0.0, ['homoglyph']),
]
# for _atk in ['all', 'article_deletion', 'homoglyph', 'paraphrase', 'whitespace', 'alternative_spelling', 'zero_width_space', 'insert_paragraphs', 'synonym', 'perplexity_misspelling', 'number', 'upper_lower'][-7:]:
for _atk in ['homoglyph']:
    EVAL_GROUPS.append((_atk, 'TEDn', 0.5, [_atk]))
    EVAL_GROUPS.append((_atk, 'PNU',  0.5, [_atk]))
    # pass

TEST_ALPHA = 0.5
TEST_CIS = [0.90, 0.95, 0.99]
N_BOOTSTRAP = 2500

OUTPUT_CSV = "logging_accuracy_raid_cross_per_seed.csv"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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
    """preds_p/preds_u/u_targets are single-element lists (one seed)."""
    preds_up_list, preds_un_list = [], []
    for i in range(len(preds_u)):
        preds_up_list.append(preds_u[i][u_targets[i] == 0][:, 0])
        preds_un_list.append(preds_u[i][u_targets[i] == 1][:, 0])

    d = {}
    for name, fn in [
        ('auc',             auc_fn),
        ('pos_prob',        pos_prob_fn),
        ('neg_prob',        neg_prob_fn),
        ('avg_pos_neg_prob', avg_prob_fn),
        ('tpr',             tpr_fn),
        ('fnr',             fnr_fn),
        ('tnr',             tnr_fn),
        ('fpr',             fpr_fn),
        ('plugin',          plugin_fn),
        ('plugin-int',      plugin_int_fn),
        ('entropy',         binary_entropy_fn),
        ('entropy_pos',     binary_entropy_pos_fn),
        ('entropy_neg',     binary_entropy_neg_fn),
        ('bce',             balanced_cross_entropy_fn),
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


# -----------------------------------------------------------------------
# Shift-based eval groups
# -----------------------------------------------------------------------
SHIFT_EVAL_GROUPS = []
for _shift_col, _src, _tgt in [
    # ('repetition_penalty', 'no',        'yes'),
    # ('decoding',           'greedy',    'sampling'),
    # ('domain',             'abstracts', 'news'),
    # ('model',              'llama-chat','chatgpt'),
]:
    SHIFT_EVAL_GROUPS.append((_shift_col, _src, _tgt, 'PN',   0.0))
    SHIFT_EVAL_GROUPS.append((_shift_col, _src, _tgt, 'TEDn', 0.5))
    # SHIFT_EVAL_GROUPS.append((_shift_col, _src, _tgt, 'PNU',  0.5))

SHIFT_OUTPUT_CSV = "logging_accuracy_raid_shift_per_seed.csv"

if os.path.exists(OUTPUT_CSV):
    metrics_df = pd.read_csv(OUTPUT_CSV)
else:
    metrics_df = pd.DataFrame()

run_id = len(metrics_df)

for train_attack, train_method, train_alpha, test_attacks in EVAL_GROUPS:
    nets = {}
    for seed in SEEDS:
        model_dir = Path(f"{RAID_LOG_BASE}/{train_attack}/{train_method}_{seed}/raid_{EPOCHS}")
        if not model_dir.exists():
            print(f"[skip] {model_dir} not found")
            continue
        pts = [p for p in model_dir.iterdir()
               if p.is_file() and p.name.lower().endswith('.pt') and train_method in p.name]
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
        print(f"[skip] no models for train_attack={train_attack} method={train_method}")
        continue

    available_seeds = sorted(nets.keys())

    for test_attack in test_attacks:
        alpha = 1.0 if test_attack == 'human' else TEST_ALPHA
        print(f"train_attack={train_attack} method={train_method} | test_attack={test_attack} alpha={alpha}")

        for seed in available_seeds:
            print(f"getting preds seed {seed}")
            pos_probs, unlabeled_probs, unlabeled_targets = get_preds_raid(
                nets[seed], device, test_attack, alpha, seed
            )
            print(f"got preds seed {seed}")
            save_preds(
                f"{PREDS_BASE}/raid/{train_attack}/{train_method}/test_{test_attack}/seed_{seed}.npz",
                pos_probs, unlabeled_probs, unlabeled_targets,
            )

            info = {
                'train_attack':  train_attack,
                'train_method':  train_method,
                'train_alpha':   train_alpha,
                'test_attack':   test_attack,
                'test_alpha':    alpha,
                'epochs':        EPOCHS,
                'seed':          seed,
                'run_id':        run_id,
            }

            metrics = get_metrics(
                [pos_probs], [unlabeled_probs], [unlabeled_targets],
                test_cis=TEST_CIS, n_bootstrap=N_BOOTSTRAP,
            )

            row = {}
            row.update(info)
            row.update(metrics)

            metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
            metrics_df.to_csv(OUTPUT_CSV, index=False)

    for net in nets.values():
        net.cpu()
    del nets
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------
# Shift-based evaluation loop
# -----------------------------------------------------------------------
if os.path.exists(SHIFT_OUTPUT_CSV):
    shift_metrics_df = pd.read_csv(SHIFT_OUTPUT_CSV)
else:
    shift_metrics_df = pd.DataFrame()

shift_run_id = len(shift_metrics_df)

print("running shift")

for shift_col, source_val, target_val, train_method, train_alpha in SHIFT_EVAL_GROUPS:
    data_type = f"raid_{shift_col}"
    llm_str = source_val if train_method == 'PN' else f"{source_val}:{target_val}"

    nets = {}
    for seed in SEEDS:
        model_dir = Path(
            f"{RAID_LOG_BASE}/{data_type}/{llm_str}/{train_method}_{seed}"
            f"/{data_type}_{EPOCHS}"
        )
        if not model_dir.exists():
            print(f"[skip] {model_dir} not found")
            continue
        pts = [p for p in model_dir.iterdir()
               if p.is_file() and p.name.lower().endswith('.pt') and train_method in p.name]
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
        print(f"[skip] no models for {data_type} {llm_str} method={train_method}")
        continue

    available_seeds = sorted(nets.keys())
    alpha = TEST_ALPHA
    print(f"shift={shift_col} {source_val}->{target_val} method={train_method} | "
          f"testing on target_val={target_val} alpha={alpha}")

    for seed in available_seeds:
        print(f"getting preds seed {seed}")
        pos_probs, unlabeled_probs, unlabeled_targets = get_preds_raid_shift(
            nets[seed], device, shift_col, target_val, alpha, seed
        )
        save_preds(
            f"{PREDS_BASE}/raid_shift/{data_type}/{llm_str}/{train_method}/seed_{seed}.npz",
            pos_probs, unlabeled_probs, unlabeled_targets,
        )

        info = {
            'shift_col':    shift_col,
            'source_val':   source_val,
            'target_val':   target_val,
            'train_method': train_method,
            'train_alpha':  train_alpha,
            'test_alpha':   alpha,
            'epochs':       EPOCHS,
            'seed':         seed,
            'run_id':       shift_run_id,
        }

        metrics = get_metrics(
            [pos_probs], [unlabeled_probs], [unlabeled_targets],
            test_cis=TEST_CIS, n_bootstrap=N_BOOTSTRAP,
        )

        row = {}
        row.update(info)
        row.update(metrics)

        shift_metrics_df = pd.concat([shift_metrics_df, pd.DataFrame([row])], ignore_index=True)
        shift_metrics_df.to_csv(SHIFT_OUTPUT_CSV, index=False)

    for net in nets.values():
        net.cpu()
    del nets
    torch.cuda.empty_cache()
