"""
Build continual-learning accuracy curves for the TTT strategies from the per-sample
records CSV written by eval_TEDn_X_rewrite_Z_ttt.py.

Figures (saved to <outdir>):
  ttt_curve_sample.png             overlay: moving-average accuracy vs sample number
  ttt_curve_sample_cumulative.png  overlay: cumulative accuracy vs sample number
  ttt_curve_batch.png              overlay: per-batch accuracy vs batch number
  ttt_curve_<strategy>_sample.png  per strategy: raw + moving-avg + cumulative

Expected shapes: online rises then plateaus (above none); none flat; episodic climbs
within each batch and drops at the reset boundary (sawtooth at the sample level, flat
at the batch level).

Standalone:  python plot_ttt_curves.py [samples_csv] [outdir]
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

STRATEGY_ORDER = ["none", "episodic", "online"]
STRATEGY_COLOR = {"none": "#7f7f7f", "episodic": "#ff7f0e", "online": "#1f77b4"}


def _rolling(series, window):
    return series.rolling(window=window, center=True, min_periods=max(1, window // 4)).mean()


def _sample_curves(df, col="correct"):
    """Mean-across-seeds `col` per sample_id, for each strategy."""
    out = {}
    for strat, g in df.groupby("strategy"):
        out[strat] = g.groupby("sample_id")[col].mean().sort_index()
    return out


def _batch_curves(df, col="correct"):
    """Mean-across-seeds per-batch `col`, for each strategy."""
    out = {}
    for strat, g in df.groupby("strategy"):
        per_seed_batch = g.groupby(["seed", "batch_id"])[col].mean().reset_index()
        out[strat] = per_seed_batch.groupby("batch_id")[col].mean().sort_index()
    return out


def _overlay_accuracy(curves, strategies, color, outpath, xlabel, ylabel, title, window):
    """Overlay smoothed accuracy curves (faint raw + bold moving-avg) for each strategy."""
    plt.figure(figsize=(9, 5))
    for s in strategies:
        ser = curves[s]
        plt.plot(ser.index, ser.values, color=color(s), alpha=0.12, linewidth=0.8)
        plt.plot(ser.index, _rolling(ser, window).values, color=color(s), linewidth=2.2, label=s)
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _present(strategies):
    return [s for s in STRATEGY_ORDER if s in strategies] + \
           [s for s in strategies if s not in STRATEGY_ORDER]


def _safe_auc(y, s):
    """AUC that higher prob_ai => AI (== the reported flip=False metric, since
    roc_auc_score(1-y,1-s)==roc_auc_score(y,s)). NaN if a window is single-class."""
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return np.nan
    return roc_auc_score(y, np.asarray(s))


def _sliding_auc(df, window, stride):
    """Per strategy: AUC over the trailing `window` samples, every `stride` samples,
    computed per seed then averaged across seeds (aligned on the right-edge sample_id)."""
    out = {}
    for strat, g in df.groupby("strategy"):
        pieces = []
        for _seed, gs in g.groupby("seed"):
            gs = gs.sort_values("sample_id")
            y = (gs["label"].values == "ai").astype(int)
            s = gs["prob_ai"].values
            sid = gs["sample_id"].values
            n = len(gs)
            idx, val = [], []
            for end in range(window, n + 1, stride):
                idx.append(sid[end - 1])
                val.append(_safe_auc(y[end - window:end], s[end - window:end]))
            pieces.append(pd.Series(val, index=idx))
        out[strat] = pd.concat(pieces, axis=1).mean(axis=1).sort_index() if pieces else pd.Series(dtype=float)
    return out


def _batch_auc(df):
    """Per strategy: AUC within each batch (per seed), averaged across seeds by batch_id."""
    out = {}
    for strat, g in df.groupby("strategy"):
        rows = []
        for (_seed, b), gb in g.groupby(["seed", "batch_id"]):
            rows.append((b, _safe_auc((gb["label"].values == "ai").astype(int), gb["prob_ai"].values)))
        pdf = pd.DataFrame(rows, columns=["batch_id", "auc"])
        out[strat] = pdf.groupby("batch_id")["auc"].mean().sort_index()
    return out


def _batch_recall(df):
    """Per strategy: AI recall (TPR @0.5) within each batch = fraction of AI items with
    pred_ai==1, per seed then averaged across seeds by batch_id."""
    out = {}
    ai = df[df["label"] == "ai"]
    for strat, g in ai.groupby("strategy"):
        per_seed_batch = g.groupby(["seed", "batch_id"])["pred_ai"].mean().reset_index()
        out[strat] = per_seed_batch.groupby("batch_id")["pred_ai"].mean().sort_index()
    return out


def make_curves(samples_csv, outdir, window=51, title_suffix="",
                auc_window=300, auc_stride=None):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(samples_csv)
    strategies = _present(df["strategy"].unique().tolist())
    if auc_stride is None:
        auc_stride = max(1, auc_window // 10)

    sample_c = _sample_curves(df)
    batch_c = _batch_curves(df)

    def color(s):
        return STRATEGY_COLOR.get(s, None)

    # ---- 1. sample-level moving average (overlay) ----
    plt.figure(figsize=(9, 5))
    for s in strategies:
        ser = sample_c[s]
        plt.plot(ser.index, ser.values, color=color(s), alpha=0.12, linewidth=0.8)
        plt.plot(ser.index, _rolling(ser, window).values, color=color(s),
                 linewidth=2.2, label=s)
    plt.xlabel("test sample number (stream order)")
    plt.ylabel(f"accuracy (moving avg, w={window})")
    plt.title(f"TTT accuracy vs sample number{title_suffix}")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ttt_curve_sample.png"), dpi=150)
    plt.close()

    # ---- 2. sample-level cumulative accuracy (overlay) ----
    plt.figure(figsize=(9, 5))
    for s in strategies:
        ser = sample_c[s]
        cum = ser.expanding().mean()
        plt.plot(ser.index, cum.values, color=color(s), linewidth=2.2, label=s)
    plt.xlabel("test sample number (stream order)")
    plt.ylabel("cumulative accuracy")
    plt.title(f"TTT cumulative accuracy vs sample number{title_suffix}")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ttt_curve_sample_cumulative.png"), dpi=150)
    plt.close()

    # ---- 3. batch-level accuracy (overlay) ----
    plt.figure(figsize=(9, 5))
    for s in strategies:
        ser = batch_c[s]
        plt.plot(ser.index, ser.values, color=color(s), marker="o", markersize=3,
                 linewidth=1.6, label=s)
    plt.xlabel("test batch number")
    plt.ylabel("per-batch accuracy")
    plt.title(f"TTT accuracy vs batch number{title_suffix}")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ttt_curve_batch.png"), dpi=150)
    plt.close()

    # ---- 3b. sliding-window AUC vs sample number ----
    auc_sample = _sliding_auc(df, window=auc_window, stride=auc_stride)
    plt.figure(figsize=(9, 5))
    for s in strategies:
        ser = auc_sample[s]
        plt.plot(ser.index, ser.values, color=color(s), linewidth=2.0, label=s)
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    plt.xlabel("test sample number (right edge of window)")
    plt.ylabel(f"AUC (sliding window, last {auc_window})")
    plt.title(f"TTT sliding-window AUC vs sample number{title_suffix}")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ttt_curve_sample_auc.png"), dpi=150)
    plt.close()

    # ---- 3c. per-batch AUC vs batch number (raw + smoothed) ----
    auc_batch = _batch_auc(df)
    plt.figure(figsize=(9, 5))
    for s in strategies:
        ser = auc_batch[s]
        plt.plot(ser.index, ser.values, color=color(s), alpha=0.20, linewidth=0.8)
        plt.plot(ser.index, _rolling(ser, max(5, window // 3)).values, color=color(s),
                 linewidth=2.0, label=s)
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    plt.xlabel("test batch number")
    plt.ylabel("per-batch AUC")
    plt.title(f"TTT per-batch AUC vs batch number{title_suffix}")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ttt_curve_batch_auc.png"), dpi=150)
    plt.close()

    # ---- 3d. per-batch AI recall (TPR @0.5) vs batch number (raw + smoothed) ----
    recall_batch = _batch_recall(df)
    plt.figure(figsize=(9, 5))
    for s in strategies:
        ser = recall_batch[s]
        plt.plot(ser.index, ser.values, color=color(s), alpha=0.20, linewidth=0.8)
        plt.plot(ser.index, _rolling(ser, max(5, window // 3)).values, color=color(s),
                 linewidth=2.0, label=s)
    plt.xlabel("test batch number")
    plt.ylabel("AI recall (TPR @0.5)")
    plt.title(f"TTT AI recall vs batch number{title_suffix}")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ttt_curve_batch_recall.png"), dpi=150)
    plt.close()

    # ---- 4. per-strategy detail (raw + moving avg + cumulative) ----
    for s in strategies:
        ser = sample_c[s]
        plt.figure(figsize=(9, 5))
        plt.plot(ser.index, ser.values, color=color(s), alpha=0.15, linewidth=0.8,
                 label="per-sample (mean over seeds)")
        plt.plot(ser.index, _rolling(ser, window).values, color=color(s),
                 linewidth=2.2, label=f"moving avg (w={window})")
        plt.plot(ser.index, ser.expanding().mean().values, color="black",
                 linewidth=1.5, linestyle="--", label="cumulative")
        plt.xlabel("test sample number (stream order)")
        plt.ylabel("accuracy")
        plt.title(f"TTT [{s}] accuracy vs sample number{title_suffix}")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"ttt_curve_{s}_sample.png"), dpi=150)
        plt.close()

    # ---- 5. P-set-calibrated accuracy (threshold from known-human text, not 0.5) ----
    if "correct_cal" in df.columns:
        _overlay_accuracy(
            _sample_curves(df, "correct_cal"), strategies, color,
            os.path.join(outdir, "ttt_curve_sample_acc_cal.png"),
            "test sample number (stream order)",
            f"calibrated accuracy (moving avg, w={window})",
            f"TTT calibrated accuracy vs sample number{title_suffix}", window)
        _overlay_accuracy(
            _batch_curves(df, "correct_cal"), strategies, color,
            os.path.join(outdir, "ttt_curve_batch_acc_cal.png"),
            "test batch number", "calibrated per-batch accuracy",
            f"TTT calibrated accuracy vs batch number{title_suffix}", max(5, window // 3))

    print(f"wrote curves to {outdir} for strategies: {strategies}")


def _batch_fracz(df):
    """Fraction of a batch's AI items that are rewrite_Z, avg across seeds by batch_id."""
    ai = df[df["label"] == "ai"].copy()
    ai["isz"] = (ai["ai_source"] == "Z").astype(int)
    per = ai.groupby(["seed", "batch_id"])["isz"].mean().reset_index()
    return per.groupby("batch_id")["isz"].mean().sort_index()


def _batch_subset_auc(df, source):
    """Per-batch AUC on {human vs AI-of-`source`} only, avg across seeds. Isolates the
    model's competence on one AI source (e.g. Z) as it ramps in, free of the changing
    mixture difficulty."""
    sub = df[(df["label"] == "human") | ((df["label"] == "ai") & (df["ai_source"] == source))]
    out = {}
    for strat, g in sub.groupby("strategy"):
        rows = []
        for (_seed, b), gb in g.groupby(["seed", "batch_id"]):
            rows.append((b, _safe_auc((gb["label"].values == "ai").astype(int), gb["prob_ai"].values)))
        out[strat] = pd.DataFrame(rows, columns=["batch_id", "auc"]).groupby("batch_id")["auc"].mean().sort_index()
    return out


def _batch_subset_recall(df, source):
    """Per-batch recall (TPR@0.5) on AI-of-`source` only, avg across seeds."""
    sub = df[(df["label"] == "ai") & (df["ai_source"] == source)]
    out = {}
    for strat, g in sub.groupby("strategy"):
        per = g.groupby(["seed", "batch_id"])["pred_ai"].mean().reset_index()
        out[strat] = per.groupby("batch_id")["pred_ai"].mean().sort_index()
    return out


def make_ramp_plots(samples_csv, outdir, title_suffix=""):
    """Gradual-shift diagnostics (only when ai_source is present): AUC and recall on the
    Z-subset over batches, with the %Z schedule overlaid. The Z-subset curves isolate
    'is the model getting better at rewrite_Z' from 'the stream is getting harder'."""
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(samples_csv)
    if "ai_source" not in df.columns or (df["ai_source"] == "Z").sum() == 0:
        return  # not a gradual run, or no Z present (growth=0)
    strategies = _present(df["strategy"].unique().tolist())
    fracz = _batch_fracz(df)

    for metric, fn, ylab, fname in [
        ("auc", _batch_subset_auc, "AUC on {human vs Z-AI}", "ttt_curve_zsubset_auc.png"),
        ("recall", _batch_subset_recall, "Z-AI recall (TPR@0.5)", "ttt_curve_zsubset_recall.png"),
    ]:
        curves = fn(df, "Z")
        plt.figure(figsize=(9, 5))
        for s in strategies:
            ser = curves.get(s)
            if ser is None or ser.empty:
                continue
            w = max(5, len(ser) // 25)
            plt.plot(ser.index, _rolling(ser, w).values, color=STRATEGY_COLOR.get(s), linewidth=2.2, label=s)
        plt.plot(fracz.index, fracz.values, color="black", linestyle=":", linewidth=1.0, label="%Z in batch")
        plt.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
        plt.xlabel("test batch number")
        plt.ylabel(ylab)
        plt.title(f"Z-subset {metric} vs batch{title_suffix}")
        plt.ylim(0, 1)
        plt.legend(title="strategy")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close()
    print(f"wrote ramp Z-subset plots -> {outdir}")


def make_eda_pred_hist(samples_csv, outdir, modes=("none", "online"), title_suffix=""):
    """EDA: per mode, one figure with two panels -- FIRST batch (left) and LAST batch
    (right) of the test stream. In each panel, two overlaid histograms of P(AI):
    human writing (grey) vs AI writing (blue). One figure per mode. Log-x since
    predictions pile up near 0. Comparing left->right panel shows how the human/AI
    separation evolves as adaptation proceeds over the stream."""
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(samples_csv)
    bins = np.logspace(-5, 0, 41)
    for mode in modes:
        if mode not in df["strategy"].unique():
            continue
        g = df[df["strategy"] == mode]
        bmin, bmax = int(g["batch_id"].min()), int(g["batch_id"].max())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        for ax, bid, name in [(axes[0], bmin, f"first batch (id {bmin})"),
                              (axes[1], bmax, f"last batch (id {bmax})")]:
            b = g[g["batch_id"] == bid]
            hp = np.clip(b.loc[b["label"] == "human", "prob_ai"].values, 1e-5, 1)
            ap = np.clip(b.loc[b["label"] == "ai", "prob_ai"].values, 1e-5, 1)
            ax.hist(hp, bins=bins, alpha=0.55, color="#7f7f7f", label="human writing")
            ax.hist(ap, bins=bins, alpha=0.55, color="#1f77b4", label="AI writing")
            ax.axvline(0.5, color="red", linestyle=":", linewidth=1, label="0.5 threshold")
            ax.set_xscale("log")
            ax.set_xlabel("P(AI)")
            ax.set_title(name)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("count")
        fig.suptitle(f"{mode}: P(AI) for human vs AI writing, first vs last batch{title_suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"eda_pred_hist_{mode}.png"), dpi=150)
        plt.close()
        print(f"wrote {outdir}/eda_pred_hist_{mode}.png")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "logging_accuracy_xz_TTT_rewrite_Z_samples.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "figs/ttt"
    make_curves(csv, out)
