"""Fast-DetectGPT zero-shot baseline rows for the LLM-detection heatmap.

Stage 3 of 3 (see dump_fastdetect_texts.py). Turns the cached curvature scores into the
exact (pos_probs, unlabeled_probs, unlabeled_targets) contract that get_metrics expects, and
writes rows in the same schema as prepare_heatmap.py / prepare_heatmap_codex.py.

Fast-DetectGPT emits an unbounded score d(x) with HIGH = machine, so P(human) needs a
monotone *decreasing* map. AUC is invariant to that map, but every thresholded metric
(tpr/fpr/plugin) and every probability-valued metric (pos_prob, entropy, bce, bbe) is not.
So each cell is emitted three ways:

  FastDetectGPT-raw     P(human) = sigmoid(-d)           out-of-the-box zero-shot
  FastDetectGPT-platt   P(human) = sigmoid(A*(-d) + B)   (A,B) fit on the SOURCE llm only,
                                                         then frozen across all test llms
                                                         -- the fair non-adaptive baseline
  FastDetectGPT-oracle  threshold maximising balanced    upper bound; shows whether the
                        accuracy on the test set itself  degradation is more than threshold drift

Only `platt` depends on train_llm; raw/oracle do not (train_llm = "NA").

Usage (env: /home/kkr36/.conda/envs/llm_embeddings):
    python prepare_heatmap_fastdetect.py --granularity sentence --model EleutherAI/gpt-neo-2.7B
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# NOTE: we deliberately do NOT `from prepare_heatmap import ...`. prepare_heatmap.py has no
# __main__ guard, so importing it executes its entire eval loop (loading DistilBert models and
# running forward passes) -- which both wastes work and, on older CPU nodes, SIGILLs under
# torch 2.8. Instead we replicate its small metric helpers locally, exactly as
# prepare_heatmap_conda.py does. The only shared deps are the torch-free bootstrap functions.
from prepare_metrics import (
    bootstrap_metric, bootstrap_metric_bbe,
    auc_fn, pos_prob_fn, neg_prob_fn, avg_prob_fn,
    tpr_fn, fnr_fn, tnr_fn, fpr_fn,
    plugin_fn, plugin_int_fn,
    binary_entropy_fn, binary_entropy_pos_fn, binary_entropy_neg_fn,
    balanced_cross_entropy_fn,
)
from estimator import BBE_estimator  # pure-numpy estimator
from fastdetect import ScoreCache, TEXTS_BASE, model_slug, cache_slug, config_identity

PREDS_BASE = "/share/garg/arxiv_kaggle/predictions"


def save_preds(path, pos_probs, unlabeled_probs, unlabeled_targets):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, pos_probs=pos_probs, unlabeled_probs=unlabeled_probs,
                        unlabeled_targets=unlabeled_targets)


def _update_dict(d, metric, point, lowers, uppers):
    d[metric] = point
    for ci in uppers:
        assert ci in lowers
        d[f'{metric}_l_{ci}'] = lowers[ci]
        d[f'{metric}_u_{ci}'] = uppers[ci]


def get_metrics(preds_p, preds_u, u_targets, test_cis, n_bootstrap, canon_u=None):
    """Metric bundle mirroring prepare_heatmap.get_metrics (copied to avoid importing that module).

    canon_u: optional list of (Nu,2) canonical P(human) arrays used *only* for AUC. AUC is
    rank-based and calibration-free, so it must be identical across the raw/platt/oracle maps.
    But the source-fitted Platt logistic can learn a negative slope (flipping the decision
    direction for an inverted source like Codex), and sigmoid saturation breaks ties differently
    per map -- both of which would corrupt an AUC computed from the variant's own probabilities.
    Passing the fixed canonical score here keeps `auc` a clean separability measure.
    """
    preds_up_list, preds_un_list = [], []
    for i in range(len(preds_u)):
        preds_up_list.append(preds_u[i][u_targets[i] == 0][:, 0])
        preds_un_list.append(preds_u[i][u_targets[i] == 1][:, 0])

    if canon_u is None:
        canon_u = preds_u
    canon_up_list, canon_un_list = [], []
    for i in range(len(canon_u)):
        canon_up_list.append(canon_u[i][u_targets[i] == 0][:, 0])
        canon_un_list.append(canon_u[i][u_targets[i] == 1][:, 0])

    md = {}
    print('calculating metrics')
    _update_dict(md, "auc", *bootstrap_metric(auc_fn, canon_up_list, canon_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "pos_prob", *bootstrap_metric(pos_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "neg_prob", *bootstrap_metric(neg_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "avg_pos_neg_prob", *bootstrap_metric(avg_prob_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "tpr", *bootstrap_metric(tpr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "fnr", *bootstrap_metric(fnr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "tnr", *bootstrap_metric(tnr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "fpr", *bootstrap_metric(fpr_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "plugin", *bootstrap_metric(plugin_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "plugin-int", *bootstrap_metric(plugin_int_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "entropy", *bootstrap_metric(binary_entropy_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "entropy_pos", *bootstrap_metric(binary_entropy_pos_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "entropy_neg", *bootstrap_metric(binary_entropy_neg_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "bce", *bootstrap_metric(balanced_cross_entropy_fn, preds_up_list, preds_un_list, n_bootstrap=n_bootstrap, cis=test_cis))
    _update_dict(md, "bbe", *bootstrap_metric_bbe(BBE_estimator, preds_p, preds_u, u_targets, n_bootstrap=n_bootstrap, cis=test_cis))
    return md

DATA_TYPE = "ArXiv_BERT"
TRAIN_YEAR = 2020
TEST_YEAR = 2020
TEST_ALPHA = 0.5
CLEAN = True
GEMINI = False
EVAL_FLIP = True
SEEDS = 5
TEST_CIS = [.9, .95, .99]
N_BOOTSTRAP = 2500

ORIG_LLMS = ["Qwen", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct"]
CODEX = "Codex"
ALL_LLMS = ORIG_LLMS + [CODEX]

VARIANTS = ["FastDetectGPT-raw", "FastDetectGPT-platt", "FastDetectGPT-oracle"]


def slug(name):
    return name.replace(" ", "_")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_eval(granularity, test_llm, seed):
    path = os.path.join(TEXTS_BASE, granularity, "eval", slug(test_llm), f"seed_{seed}.json")
    with open(path) as f:
        return json.load(f)


def load_calib(granularity, source_llm, seed):
    path = os.path.join(TEXTS_BASE, granularity, "calib", slug(source_llm), f"seed_{seed}.json")
    with open(path) as f:
        return json.load(f)


def fit_source_platt(cache, granularity, source_llm, seed):
    """1-D logistic on the source LLM's own labeled calib split; frozen for all test llms.

    Fit on the *negated* score so the learned map is P(human) directly. platt_scaling.py's
    fit_platt_scaler wants a torch classifier + DataLoader, so it is the same model but the
    wrong interface for a scalar score -- sklearn is the direct route.
    """
    blob = load_calib(granularity, source_llm, seed)
    d_human = cache.lookup(blob["human_texts"])
    d_llm = cache.lookup(blob["llm_texts"])

    x = np.concatenate([-d_human, -d_llm]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(d_human)), np.zeros(len(d_llm))])  # 1 = human

    ok = np.isfinite(x).ravel()
    x, y = x[ok], y[ok]
    assert len(x) > 100, f"too little calib data for {source_llm} seed {seed}: {len(x)}"

    lr = LogisticRegression(max_iter=1000)
    lr.fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def oracle_threshold(d, is_human):
    """Threshold on -d maximising balanced accuracy, using the test set's own labels."""
    x = -d
    order = np.argsort(x)
    cands = np.unique(x[order])
    if len(cands) > 2000:  # thin out; balanced accuracy is smooth in the threshold
        cands = np.quantile(x, np.linspace(0, 1, 2000))

    n_h, n_m = is_human.sum(), (~is_human).sum()
    best_t, best_ba = cands[0], -1.0
    for t in cands:
        pred_h = x >= t
        tpr = (pred_h & is_human).sum() / max(n_h, 1)
        tnr = ((~pred_h) & (~is_human)).sum() / max(n_m, 1)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba, best_t = ba, t
    return float(best_t)


def probs_from_scores(d, variant, platt_ab=None, oracle_t=None, scale=None):
    """Map curvature score -> P(human). All three maps are monotone decreasing in d."""
    x = -d
    if variant == "FastDetectGPT-raw":
        return sigmoid(x)
    if variant == "FastDetectGPT-platt":
        a, b = platt_ab
        return sigmoid(a * x + b)
    if variant == "FastDetectGPT-oracle":
        # centre the sigmoid on the oracle threshold so 0.5 lands exactly there
        return sigmoid((x - oracle_t) / scale)
    raise ValueError(variant)


def build_cell(cache, granularity, test_llm, seed):
    """Scores + labels for one (test_llm, seed), in the get_metrics contract's polarity."""
    blob = load_eval(granularity, test_llm, seed)

    d_p = cache.lookup(blob["p_texts"])
    d_u = cache.lookup(blob["u_texts"])
    # u_labels is 1=human; UnlabelData.true_targets (helper.py) inverts it and get_metrics
    # slices positives with u_targets == 0, so the contract needs 1 - u_labels.
    targets = 1 - np.asarray(blob["u_labels"], dtype=int)

    ok_p = np.isfinite(d_p)
    ok_u = np.isfinite(d_u)
    if (~ok_p).any() or (~ok_u).any():
        print(f"    dropping {int((~ok_p).sum())} P / {int((~ok_u).sum())} U non-finite scores")
    return d_p[ok_p], d_u[ok_u], targets[ok_u]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["sentence", "abstract"], default="sentence")
    ap.add_argument("--model", default="EleutherAI/gpt-neo-2.7B", help="scoring model (p)")
    ap.add_argument("--sampling-model", default=None,
                    help="sampling model (q) for the two-model analytic pair; must match the "
                         "--sampling-model used when the scores were cached by fastdetect.py")
    ap.add_argument("--llms", nargs="*", default=ALL_LLMS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    output_csv = args.output or f"logging_accuracy_llm_fastdetect_{args.granularity}.csv"
    metrics_df = pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()

    cache = ScoreCache(cache_slug(args.model, args.sampling_model), args.granularity)
    ref_model = config_identity(args.model, args.sampling_model)
    ref_slug = cache_slug(args.model, args.sampling_model)  # path-safe for preds dirs

    # Platt params are per (source_llm, seed) and reused across every test column -- that
    # reuse is precisely what makes this baseline non-adaptive.
    platt = {}
    if "FastDetectGPT-platt" in args.variants:
        for source_llm in args.llms:
            for seed in range(args.seeds):
                platt[(source_llm, seed)] = fit_source_platt(cache, args.granularity, source_llm, seed)
                a, b = platt[(source_llm, seed)]
                print(f"platt {source_llm} seed {seed}: A={a:.4f} B={b:.4f}")

    def already_done(method, train_llm, test_llm):
        if len(metrics_df) == 0:
            return False
        m = ((metrics_df["learning_method"] == method)
             & (metrics_df["train_llm"].astype(str) == str(train_llm))
             & (metrics_df["test_llm"] == test_llm)
             & (metrics_df["ref_model"] == ref_model)
             & (metrics_df["granularity"] == args.granularity))
        return bool(m.any())

    for test_llm in args.llms:
        # score once per (test_llm, seed); every variant reuses these
        cells = [build_cell(cache, args.granularity, test_llm, seed) for seed in range(args.seeds)]

        # Curvature is expected higher for LLM text than human. When it is not, the score is
        # anti-correlated for this generator (a real non-robustness finding, e.g. Codex), NOT a
        # code bug -- the abstract-level gate confirms polarity is correct where the score works.
        for seed, (d_p, d_u, tgt) in enumerate(cells):
            m_h, m_m = d_u[tgt == 0].mean(), d_u[tgt == 1].mean()
            flag = "" if m_m > m_h else "   <-- score ANTI-CORRELATED for this LLM (d_llm < d_human)"
            print(f"  {test_llm} seed {seed}: mean d(human)={m_h:.4f} mean d(llm)={m_m:.4f}{flag}")

        for variant in args.variants:
            sources = args.llms if variant == "FastDetectGPT-platt" else ["NA"]

            for source_llm in sources:
                if already_done(variant, source_llm, test_llm):
                    print(f"skip (present): {variant} {source_llm} -> {test_llm}")
                    continue

                print(f"{variant}: train={source_llm} | test={test_llm} | ref={ref_model}")

                preds_p_list, preds_u_list, u_targets_list, canon_u_list = [], [], [], []
                for seed, (d_p, d_u, tgt) in enumerate(cells):
                    kw = {}
                    if variant == "FastDetectGPT-platt":
                        kw["platt_ab"] = platt[(source_llm, seed)]
                    elif variant == "FastDetectGPT-oracle":
                        kw["oracle_t"] = oracle_threshold(d_u, tgt == 0)
                        # scale from the test set's own spread so probabilities aren't degenerate
                        kw["scale"] = max(np.std(-d_u), 1e-6)

                    p_h = probs_from_scores(d_p, variant, **kw)
                    u_h = probs_from_scores(d_u, variant, **kw)
                    u_2d = np.stack([u_h, 1.0 - u_h], axis=1)
                    # canonical (calibration-free) score for AUC: raw -d, fixed high-d=machine.
                    # col 0 stands in for P(human); only its ranking is used by auc_fn.
                    canon_2d = np.stack([-d_u, d_u], axis=1)

                    save_preds(
                        f"{PREDS_BASE}/heatmap_fastdetect/{ref_slug}/"
                        f"{args.granularity}/{variant}/train_{slug(str(source_llm))}/"
                        f"test_{slug(test_llm)}/seed_{seed}.npz",
                        p_h, u_2d, tgt,
                    )
                    preds_p_list.append(p_h)
                    preds_u_list.append(u_2d)
                    u_targets_list.append(tgt)
                    canon_u_list.append(canon_2d)

                info = {
                    "learning_method": variant,
                    "data_type": DATA_TYPE,
                    "ref_model": ref_model,
                    "granularity": args.granularity,
                    "train_alpha": np.nan,
                    "train_year": TRAIN_YEAR,
                    "train_llm": source_llm,
                    "test_alpha": TEST_ALPHA,
                    "test_year": TEST_YEAR,
                    "test_llm": test_llm,
                    "epochs": np.nan,
                    "clean": CLEAN,
                    "sentence": args.granularity == "sentence",
                    "gemini": GEMINI,
                    "eval_flip": EVAL_FLIP,
                    "model_path": ref_model,
                }
                metrics = get_metrics(preds_p_list, preds_u_list, u_targets_list,
                                      test_cis=TEST_CIS, n_bootstrap=args.n_bootstrap,
                                      canon_u=canon_u_list)

                row = {}
                row.update(info)
                row.update(metrics)
                metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
                metrics_df.to_csv(output_csv, index=False)  # crash-safe

    print(f"done -> {output_csv} ({len(metrics_df)} rows)")


if __name__ == "__main__":
    main()
