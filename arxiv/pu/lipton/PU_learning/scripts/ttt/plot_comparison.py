"""
Objective-comparison analysis for the g100 rewrite_Z_332 stream.

Answers the research question: does a TASK-ALIGNED adaptation signal (classifier
entropy minimization) raise the human-vs-AI *separation* (the Pareto / ROC curve),
rather than merely sliding the operating point the way MLM does?

The decisive view is threshold-free / matched-operating-point, NOT fixed-0.5 accuracy:
  - AUC per arm x mode (does discrimination actually improve?).
  - Human recall at MATCHED AI-recall operating points (the Pareto curve). An arm that
    genuinely adapts sits ABOVE the frozen baseline here; a pure bias shift sits ON it.

Arms compared (all g100, batch adapt granularity, rewrite_Z_332, sentence-level):
  reference : MLM,     scope=trunk,     lr=1e-2   (existing z332_ramphlr_sent_g100)
  mlm_ln    : MLM,     scope=layernorm, lr=1e-3   (tag cmp_mlm_ln)
  ent_ln    : entropy, scope=layernorm, lr=1e-3   (tag cmp_ent_ln)

  python scripts/ttt/plot_comparison.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

CSV_DIR = "ttt_logging/csv"
OUT_DIR = "figs/ttt/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# (label, tag, colour) -- tag maps to logging_accuracy_xz_TTT_rewrite_Z_samples_<tag>.csv
ARMS = [
    ("MLM trunk lr1e-2 (ref)", "z332_ramphlr_sent_g100", "#B07AA1"),
    ("MLM  LN  lr1e-3",        "cmp_mlm_ln",             "#4E79A7"),
    ("Entropy LN lr1e-3",      "cmp_ent_ln",             "#E15759"),
]
AI_TARGETS = [0.60, 0.70, 0.78, 0.85]


def load(tag):
    p = f"{CSV_DIR}/logging_accuracy_xz_TTT_rewrite_Z_samples_{tag}.csv"
    if not os.path.exists(p):
        print(f"  [missing] {p} (job not finished / not merged?)")
        return None
    return pd.read_csv(p)


def auc_mean_std(sub):
    aucs = []
    for sd in sorted(sub.seed.unique()):
        s = sub[sub.seed == sd]
        y = (s.label == "ai").astype(int)
        aucs.append(roc_auc_score(y, s.prob_ai))
    return float(np.mean(aucs)), float(np.std(aucs))


def human_recall_at_ai(sub, targets):
    """Human recall (correctly-kept-human rate) at thresholds that hit each AI-recall
    target, pooled over seeds. Returns dict target -> human_recall."""
    ai = sub[sub.label == "ai"].prob_ai.values
    hu = sub[sub.label == "human"].prob_ai.values
    out = {}
    for t in targets:
        thr = np.quantile(ai, 1 - t)      # threshold catching fraction t of AI
        out[t] = float((hu < thr).mean())  # human below thr => correctly human
    return out


def roc_pareto(sub, n=200):
    """Human-recall vs AI-recall tradeoff curve (pooled seeds)."""
    ai = sub[sub.label == "ai"].prob_ai.values
    hu = sub[sub.label == "human"].prob_ai.values
    ai_rec = np.linspace(0.02, 0.98, n)
    thr = np.quantile(ai, 1 - ai_rec)
    hum_rec = np.array([(hu < t).mean() for t in thr])
    return ai_rec, hum_rec


def main():
    frames = {}
    for label, tag, _ in ARMS:
        df = load(tag)
        if df is not None:
            frames[label] = df
    if not frames:
        print("no arms loaded; run run_comparison.sh first.")
        return

    # ---- summary table (AUC + matched-operating-point human recall), online mode ----
    rows = []
    for label, tag, _ in ARMS:
        if label not in frames:
            continue
        df = frames[label]
        for mode in ["none", "online"]:
            sub = df[df.strategy == mode]
            if sub.empty:
                continue
            a_m, a_s = auc_mean_std(sub)
            hr = human_recall_at_ai(sub, AI_TARGETS)
            row = {"arm": label, "mode": mode, "auc": round(a_m, 4), "auc_std": round(a_s, 4)}
            row.update({f"hRec@AI{int(t*100)}": round(hr[t], 4) for t in AI_TARGETS})
            rows.append(row)
    summ = pd.DataFrame(rows)
    summ_path = f"{CSV_DIR}/comparison_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(summ.to_string(index=False))
    print(f"\n-> {summ_path}")

    # ---- FIG 1: Pareto / ROC overlay (human recall vs AI recall) ----
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    # single frozen baseline (identical across arms); take it from the first available arm
    base_df = next(iter(frames.values()))
    none_sub = base_df[base_df.strategy == "none"]
    if not none_sub.empty:
        x, y = roc_pareto(none_sub)
        ax.plot(x, y, color="#888888", lw=2, ls="--", label="frozen (none)")
    for label, tag, col in ARMS:
        if label not in frames:
            continue
        sub = frames[label][frames[label].strategy == "online"]
        if sub.empty:
            continue
        x, y = roc_pareto(sub)
        ax.plot(x, y, color=col, lw=2, label=f"{label} (online)")
    for t in AI_TARGETS:
        ax.axvline(t, color="#DDDDDD", lw=0.8, zorder=0)
    ax.set_xlabel("AI recall (operating point)")
    ax.set_ylabel("Human recall")
    ax.set_title("Pareto tradeoff on g100 rewrite_Z_332 (online)\nabove the frozen curve = genuine adaptation")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_xlim(0.5, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pareto_roc.pdf")
    plt.close(fig)

    # ---- FIG 2: human recall at matched AI-recall targets (grouped bars) ----
    online = summ[summ["mode"] == "online"]
    if not online.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        arms_present = list(online["arm"])
        x = np.arange(len(AI_TARGETS))
        w = 0.8 / max(len(arms_present), 1)
        colmap = {label: col for label, _, col in ARMS}
        # frozen baseline reference line per target
        base_hr = human_recall_at_ai(none_sub, AI_TARGETS) if not none_sub.empty else None
        for i, arm in enumerate(arms_present):
            vals = [online[online.arm == arm][f"hRec@AI{int(t*100)}"].iloc[0] for t in AI_TARGETS]
            ax.bar(x + i * w, vals, w, label=arm, color=colmap.get(arm, "#666"))
        if base_hr is not None:
            for j, t in enumerate(AI_TARGETS):
                ax.hlines(base_hr[t], x[j] - 0.1, x[j] + 0.8, color="#333", lw=1.4,
                          ls=":", label="frozen" if j == 0 else None)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([f"AI recall {int(t*100)}%" for t in AI_TARGETS])
        ax.set_ylabel("Human recall at matched AI recall")
        ax.set_title("Human recall at matched operating points (higher = better; > frozen = real gain)")
        ax.set_ylim(0.6, 1.0)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/human_recall_at_matched_ai.pdf")
        plt.close(fig)

    print(f"wrote pareto_roc.pdf + human_recall_at_matched_ai.pdf -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
