"""
Evaluate PN_TTT-trained DistilBertTTT models on the rewrite_Z unlabeled stream,
comparing test-time-training modes {none, episodic, online}.

Mirrors eval_TEDn_X_rewrite_Z.py (same bootstrapped metric suite, same flip=True /
human-positive polarity that is correct for the class-0=AI convention), but:
  - loads DistilBertTTT (Y-shaped) checkpoints instead of DistilBert,
  - routes the U-set through model_inference_ttt.get_preds_xy_ttt with a per-mode
    ttt_cfg, so ttt_mode=none is the frozen baseline and episodic/online adapt.

Writes logging_accuracy_xz_TTT_rewrite_Z.csv (dedicated file; canonical CSVs
untouched) with an added ttt_mode column.

NOTE: this module has no __main__ guard on purpose to match the repo's other eval
scripts -- run it directly, do not import it.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch

from model_inference_ttt import get_preds_xy_ttt_stream
from model_helper import get_model
from prepare_metrics import *
from estimator import BBE_estimator
from plot_ttt_curves import make_curves


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
        assert (ci in lowers)
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


# ---------------- SWITCHES ---------------- #
# Granularity toggle. Abstract-level uses read_arxiv_split_xy's abstract branch
# (balanced human/AI whole abstracts) for training, and get_u_data_xy at abstract
# granularity for the eval stream. MUST match how the checkpoints were trained.
ABSTRACT_LEVEL = os.environ.get("TTT_ABSTRACT", "1") == "1"
gran = "abstract" if ABSTRACT_LEVEL else "sentence"
sentence = not ABSTRACT_LEVEL   # codebase convention: sentence=True => sentence-level

# `gran` locates checkpoints (PN_TTT_<gran>_<seed>/xy_<epochs>); `tag` labels the
# output CSVs + figs subdir so different experiments don't clobber each other.
tag = os.environ.get("TTT_TAG", gran)

clean = True
epochs = int(os.environ.get("TTT_EPOCHS", "3"))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
seeds = [0, 1, 2, 3, 4]
test_alpha = 0.5                # matches prepare_optimization.py (50/50 human/AI)
test_cis = [.9, .95, .99]
# PN-style model (canonical class 0 = AI = positive) -> flip=False, exactly like the
# PN configs in prepare_optimization.py. flip only sets probability polarity for the
# aggregate metrics; the U-set data itself is flip-independent (get_u_data_xy).
flip = False
eval_col = os.environ.get("TTT_EVAL_COL", "rewrite_Z")   # AI eval column (e.g. rewrite_strategy_Z_0 on the v2 parquet)

# where PN_TTT checkpoints live (see scripts/ttt/run_arxiv_ttt.py --log-dir convention)
MODEL_BASE = "/share/garg/arxiv_kaggle/ttt_models"

# all TTT logging CSVs go here (keeps the PU_learning root clean)
OUT_DIR = "ttt_logging/csv"
os.makedirs(OUT_DIR, exist_ok=True)

# Streaming eval (per-sample) is compute-heavy: each adapted sample does n_steps
# backward passes. Average curves over a few seeds; expand once you've seen runtime.
STREAM_SEEDS = [0, 1, 2]
STREAM_BATCH_SIZE = int(os.environ.get("TTT_BATCH", "32"))  # reset/grouping + adaptation batch
MOVING_AVG_WINDOW = 51          # smoothing for the sample-level curve
# P-set-calibrated decision threshold: quantile of native P(AI) on known-human text.
# 0.95 => the threshold that flags ~5% of genuine human abstracts (5% FPR operating
# point), instead of the mis-placed fixed 0.5. Applied per (mode, seed).
CAL_TAU = 0.95

# TTT config per mode (kwargs for algorithm_ttt.ttt_stream_predict).
# batch_size fixes the episodic reset boundary + batch-level grouping.
# Adaptation config (env-overridable). adapt_granularity: 'sample' (per-instance) or
# 'batch' (adapt on the whole 32-sample batch per MLM step -> cleaner gradient).
# Gradual-shift stream: growth in [0,1]. 0=all rewrite_X (train dist), 1=all rewrite_Z
# (abrupt, prior behavior), 0.5=linear ramp X->Z hitting 100% Z at the halfway batch.
# Unset => the non-gradual (single eval_col) stream.
_g = os.environ.get("TTT_GROWTH")
GROWTH = float(_g) if _g not in (None, "") else None

ADAPT_GRAN = os.environ.get("TTT_ADAPT_GRAN", "sample")
INNER_LR = float(os.environ.get("TTT_INNER_LR", "1e-3"))
N_STEPS_EP = int(os.environ.get("TTT_NSTEPS_EP", "5"))
N_STEPS_ON = int(os.environ.get("TTT_NSTEPS_ON", "1"))
N_MASK = int(os.environ.get("TTT_NMASK", "4"))
_adapt = dict(batch_size=STREAM_BATCH_SIZE, adapt_scope="trunk", optimizer_type="sgd",
              adapt_granularity=ADAPT_GRAN, n_mask_samples=N_MASK, inner_lr=INNER_LR)
ttt_modes = {
    "none":     {"mode": "none", "batch_size": STREAM_BATCH_SIZE, "adapt_granularity": ADAPT_GRAN},
    "episodic": {"mode": "episodic", **_adapt, "n_steps": N_STEPS_EP},
    "online":   {"mode": "online", **_adapt, "n_steps": N_STEPS_ON},
}


def find_checkpoint(seed):
    # train_PU_one_year saves under log_dir/<data_type>_<epochs>/ ; data_type=xy here
    pattern = f"{MODEL_BASE}/PN_TTT_{gran}_{seed}/xy_{epochs}/PN_TTT_*.pt"
    pts = sorted(glob.glob(pattern))
    assert len(pts) >= 1, f"No PN_TTT checkpoint found at {pattern}"
    return pts[-1]  # most recent if several


def load_net(seed):
    net = get_model("DistilBertTTT")
    state_dict = torch.load(find_checkpoint(seed), map_location=device)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    return net


# ---------------- LOGIC ---------------- #
# One mode per process, selected by the TTT_MODE env var, so a 3-GPU job can run
# none/episodic/online concurrently (each pinned to its own GPU via CUDA_VISIBLE_DEVICES).
# TTT_MODE unset or "all" runs all three serially (single-GPU fallback). Each mode writes
# its OWN per-mode files (no parallel write races); merge_ttt_curves.py concatenates them
# and draws the curves.
env_mode = os.environ.get("TTT_MODE", "all").strip().lower()
if env_mode in ("all", ""):
    modes_to_run = list(ttt_modes.keys())
else:
    assert env_mode in ttt_modes, f"TTT_MODE={env_mode} not in {list(ttt_modes)}"
    modes_to_run = [env_mode]

print(f"running modes {modes_to_run} on device={device} "
      f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

for ttt_name in modes_to_run:
    ttt_cfg = ttt_modes[ttt_name]
    samples_mode_csv = f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_samples_{tag}_{ttt_name}.csv"
    metrics_mode_csv = f"{OUT_DIR}/logging_accuracy_xz_TTT_rewrite_Z_metrics_{tag}_{ttt_name}.csv"
    print(f"\n===== ttt_mode={ttt_name} on {eval_col} ({gran}-level) =====")

    preds_p_list, preds_u_list, u_targets_list = [], [], []
    mode_records = []
    for seed in STREAM_SEEDS:
        net = load_net(seed)
        pos_probs, unlabeled_probs, unlabeled_targets, records_df, pset_pai = get_preds_xy_ttt_stream(
            net, device, test_alpha, flip, seed, sentence, clean, eval_col, ttt_cfg, growth=GROWTH)
        save_preds(
            f"{PREDS_BASE}/ttt/{ttt_name}/PN_TTT/{eval_col}/seed_{seed}.npz",
            pos_probs, unlabeled_probs, unlabeled_targets,
        )
        preds_p_list.append(pos_probs)
        preds_u_list.append(unlabeled_probs)
        u_targets_list.append(unlabeled_targets)

        # Two label-free calibrated thresholds in native P(AI) space:
        #  - prevalence-matched (headline): median of the stream's P(AI); since the test
        #    set is 50/50 (test_alpha=0.5) this is the balanced operating point, so
        #    accuracy tracks AUC. Used for correct_cal + the calibrated-accuracy curves.
        #  - P-set 5% FPR: CAL_TAU-quantile of P(AI) on known-human text (strict detector
        #    operating point) -> honest "how much AI is actually caught" number.
        thr_bal = float(np.median(records_df["prob_ai"].values))
        thr_fpr = float(np.quantile(pset_pai, CAL_TAU))
        is_ai = (records_df["target"].values == 1).astype(int)
        records_df["cal_threshold"] = thr_bal
        records_df["pred_ai_cal"] = (records_df["prob_ai"].values > thr_bal).astype(int)
        records_df["correct_cal"] = (records_df["pred_ai_cal"].values == is_ai).astype(int)
        records_df["correct_cal_fpr5"] = (
            (records_df["prob_ai"].values > thr_fpr).astype(int) == is_ai).astype(int)

        records_df.insert(0, "strategy", ttt_name)
        records_df.insert(1, "seed", seed)
        records_df["eval_llm"] = eval_col
        records_df["granularity"] = gran
        mode_records.append(records_df)
        # incrementally persist so progress survives a crash
        pd.concat(mode_records, ignore_index=True).to_csv(samples_mode_csv, index=False)

    metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list,
                          test_cis=test_cis, n_bootstrap=2500)
    recs = pd.concat(mode_records, ignore_index=True)
    metrics["stream_accuracy"] = recs["correct"].mean()                 # fixed 0.5 threshold
    metrics["cal_stream_accuracy"] = recs["correct_cal"].mean()         # prevalence-matched threshold
    metrics["cal_acc_fpr5"] = recs["correct_cal_fpr5"].mean()           # P-set 5% FPR threshold
    metrics["cal_threshold_mean"] = recs.groupby("seed")["cal_threshold"].first().mean()
    info = {
        "learning_method": "PN_TTT", "ttt_mode": ttt_name, "data_type": "xy",
        "eval_llm": eval_col, "granularity": gran, "test_alpha": test_alpha,
        "flip": flip, "clean": clean, "sentence": sentence, "epochs": epochs,
        "ttt_cfg": str(ttt_cfg), "n_seeds": len(STREAM_SEEDS),
    }
    row = {}
    row.update(info)
    row.update(metrics)
    pd.DataFrame([row]).to_csv(metrics_mode_csv, index=False)
    print(f"ttt_mode={ttt_name}: AUC={metrics['auc']:.4f} | "
          f"stream_acc={metrics['stream_accuracy']:.4f} -> {samples_mode_csv}")

print(f"\nDone modes {modes_to_run}. Run merge_ttt_curves.py to combine + plot.")
