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

# Presentation-readable defaults applied to every figure below (no data changes).
plt.rcParams.update({
    "figure.figsize": (11, 7),
    "font.size": 20,
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "legend.title_fontsize": 24,
})

def _savefig(outdir, basename):
    """Save the current figure as PNG+PDF, or PDF-only when TTT_PDF_ONLY=1."""
    if os.environ.get("TTT_PDF_ONLY", "") != "1":
        plt.savefig(os.path.join(outdir, f"{basename}.png"), dpi=150)
    plt.savefig(os.path.join(outdir, f"{basename}.pdf"))

STRATEGY_ORDER = ["none", "episodic", "online"]
STRATEGY_COLOR = {"none": "#7f7f7f", "episodic": "#ff7f0e", "online": "#1f77b4"}
# Legend display names (used only when TTT_RENAME=1; unknown strategies fall back to raw name).
STRATEGY_LABEL = {"none": "Supervised",
                  "online": "TTT",
                  "episodic": "Episodic Test-Time Training"}
# Strategies to omit from every figure (comma list, e.g. TTT_EXCLUDE=episodic).
EXCLUDE = set(x for x in os.environ.get("TTT_EXCLUDE", "").split(",") if x)
RENAME = os.environ.get("TTT_RENAME", "") == "1"


def _disp(s):
    """Legend label for strategy `s`."""
    return STRATEGY_LABEL.get(s, s)


def _rolling(series, window):
    return series.rolling(window=window, center=True, min_periods=max(1, window // 4)).mean()


def _xvals(index, scale=1):
    return np.asarray(index, dtype=float) * scale


def _maybe_shift_marker(x):
    if x is not None:
        plt.axvline(x, color="red", linewidth=5.6, linestyle=":", alpha=0.9)


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


def _batch_balanced_accuracy(df, col="correct"):
    """Per strategy: balanced accuracy per batch = mean(human-side `col`, AI-side `col`),
    each averaged across seeds first. Unlike a plain per-batch mean of `col` (_batch_curves),
    this is robust to a batch's human:AI row ratio not being 1:1 -- which happens in the
    z332_only view, where AI rows are filtered down to one ai_source and can be sparse or
    absent in some batches. Batches missing either class are NaN (dropped), matching the
    existing convention for Z-subset curves (see make_ramp_plots)."""
    out = {}
    for strat, g in df.groupby("strategy"):
        def _side(label):
            sub = g[g["label"] == label]
            per_seed = sub.groupby(["seed", "batch_id"])[col].mean().reset_index()
            return per_seed.groupby("batch_id")[col].mean()
        combined = pd.concat([_side("human").rename("h"), _side("ai").rename("a")], axis=1)
        out[strat] = ((combined["h"] + combined["a"]) / 2.0).dropna().sort_index()
    return out


def _balanced_rolling(df, col, window):
    """Per strategy: rolling-window balanced accuracy vs sample number. A centered rolling
    mean of `col` is computed separately within the human-only and AI-only sub-sequences
    (each in their own sample_id order), then both are reindexed onto the full sample_id
    axis (nearest available value) and averaged. This is what makes it 'balanced': the
    result doesn't depend on the human:AI row ratio inside a window, unlike a plain rolling
    mean of `col` over all rows (relevant for the z332_only view's sparse AI rows)."""
    out = {}
    all_sids = np.sort(df["sample_id"].unique())
    for strat, g in df.groupby("strategy"):
        hu = g[g["label"] == "human"].groupby("sample_id")[col].mean().sort_index()
        ai = g[g["label"] == "ai"].groupby("sample_id")[col].mean().sort_index()
        hu_r = _rolling(hu, window).reindex(all_sids).ffill().bfill()
        ai_r = _rolling(ai, window).reindex(all_sids).ffill().bfill()
        out[strat] = ((hu_r + ai_r) / 2.0).dropna()
    return out


def _cumulative_balanced_accuracy(df, col="correct"):
    """Per strategy: cumulative balanced accuracy vs sample number = running mean of `col`
    within human rows and within AI rows so far, averaged; per seed then averaged across
    seeds by sample_id. NaN before the first row of either class has appeared in that seed's
    stream (e.g. the z332_only view before any AI-of-source row shows up)."""
    out = {}
    for strat, g in df.groupby("strategy"):
        pieces = []
        for _seed, gs in g.groupby("seed"):
            gs = gs.sort_values("sample_id")
            is_ai = (gs["label"].values == "ai")
            val = gs[col].values.astype(float)
            sid = gs["sample_id"].values
            h_cnt = np.cumsum(~is_ai)
            a_cnt = np.cumsum(is_ai)
            h_sum = np.cumsum(np.where(~is_ai, val, 0.0))
            a_sum = np.cumsum(np.where(is_ai, val, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                h_acc = np.where(h_cnt > 0, h_sum / np.maximum(h_cnt, 1), np.nan)
                a_acc = np.where(a_cnt > 0, a_sum / np.maximum(a_cnt, 1), np.nan)
            pieces.append(pd.Series((h_acc + a_acc) / 2.0, index=sid))
        out[strat] = pd.concat(pieces, axis=1).mean(axis=1).sort_index() if pieces else pd.Series(dtype=float)
    return out


def _overlay_accuracy(curves, strategies, color, outdir, basename, xlabel, ylabel, title, window,
                       bold_curves=None, x_scale=1, marker_x=None):
    """Overlay smoothed accuracy curves (faint raw + bold moving-avg) for each strategy.
    `curves` is plotted faint+raw; the bold line is `_rolling(curves[s], window)` unless
    `bold_curves` is given (already-computed per-strategy series), used when the bold line
    needs different windowing logic than the raw one (e.g. balanced accuracy)."""
    plt.figure(figsize=(11, 7))
    for s in strategies:
        ser = curves[s]
        plt.plot(_xvals(ser.index, x_scale), ser.values, color=color(s), alpha=0.12, linewidth=3.2)
        bold = bold_curves[s] if bold_curves is not None else _rolling(ser, window)
        plt.plot(_xvals(bold.index, x_scale), bold.values, color=color(s), linewidth=8.8, label=_disp(s))
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    _maybe_shift_marker(marker_x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(outdir, basename)
    plt.close()


def _present(strategies):
    ordered = [s for s in STRATEGY_ORDER if s in strategies] + \
              [s for s in strategies if s not in STRATEGY_ORDER]
    return [s for s in ordered if s not in EXCLUDE]


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


def _batch_meanprob_ai(df):
    """Per strategy: per-batch mean P(AI) over AI rows, avg across seeds by batch_id."""
    out = {}
    ai = df[df["label"] == "ai"]
    for strat, g in ai.groupby("strategy"):
        per = g.groupby(["seed", "batch_id"])["prob_ai"].mean().reset_index()
        out[strat] = per.groupby("batch_id")["prob_ai"].mean().sort_index()
    return out


def _batch_meanprob_human(df):
    """Per strategy: per-batch mean P(human)=1-P(AI) over human rows, avg across seeds.
    Independent of the AI mixture, so identical for the full stream and any AI-subset view."""
    out = {}
    hu = df[df["label"] == "human"].copy()
    hu["p_human"] = 1.0 - hu["prob_ai"]
    for strat, g in hu.groupby("strategy"):
        per = g.groupby(["seed", "batch_id"])["p_human"].mean().reset_index()
        out[strat] = per.groupby("batch_id")["p_human"].mean().sort_index()
    return out


def _batch_human_recall(df):
    """Per strategy: per-batch fraction of human rows with P(human)>=0.5 (i.e. prob_ai<=0.5),
    the human-side mirror of _batch_recall; avg across seeds. Human-only, so subset-independent."""
    out = {}
    hu = df[df["label"] == "human"].copy()
    hu["h_correct"] = ((1.0 - hu["prob_ai"]) >= 0.5).astype(int)
    for strat, g in hu.groupby("strategy"):
        per = g.groupby(["seed", "batch_id"])["h_correct"].mean().reset_index()
        out[strat] = per.groupby("batch_id")["h_correct"].mean().sort_index()
    return out


def make_curves(samples_csv, outdir, window=51, title_suffix="",
                auc_window=300, auc_stride=None, batch_size=None, mark_shift=False):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(samples_csv)
    strategies = _present(df["strategy"].unique().tolist())
    if auc_stride is None:
        auc_stride = max(1, auc_window // 10)
    batch_x_scale = batch_size if batch_size is not None else 1
    batch_xlabel = "Total # Test-Time Data Seen" if batch_size is not None else "Test Batch #"
    n_batches = int(df["batch_id"].max()) + 1
    shift_marker_x = (n_batches * batch_x_scale / 2.0) if mark_shift and batch_size is not None else None
    sample_shift_marker_x = (n_batches * batch_x_scale / 2.0) if mark_shift else None

    sample_c = _sample_curves(df)
    bal_sample_c = _balanced_rolling(df, "correct", window)
    cum_bal_sample_c = _cumulative_balanced_accuracy(df, "correct")
    batch_bal_c = _batch_balanced_accuracy(df, "correct")

    def color(s):
        return STRATEGY_COLOR.get(s, None)

    # ---- 1. sample-level moving average (overlay). Bold line is BALANCED accuracy
    # (mean of human-side and AI-side accuracy in the rolling window) so the curve isn't
    # biased by a window's human:AI row ratio -- relevant for the z332_only view, where AI
    # rows are filtered to one ai_source and can be sparse. Faint dots stay raw per-sample. ----
    plt.figure(figsize=(11, 7))
    for s in strategies:
        ser = sample_c[s]
        bal = bal_sample_c[s]
        plt.plot(ser.index, ser.values, color=color(s), alpha=0.12, linewidth=3.2)
        plt.plot(bal.index, bal.values, color=color(s), linewidth=8.8, label=_disp(s))
    _maybe_shift_marker(sample_shift_marker_x)
    plt.xlabel("Test Sample #")
    plt.ylabel("Balanced Accuracy")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(outdir, "ttt_curve_sample")
    plt.close()

    # ---- 2. sample-level cumulative BALANCED accuracy (overlay) ----
    plt.figure(figsize=(11, 7))
    for s in strategies:
        cum = cum_bal_sample_c[s]
        plt.plot(cum.index, cum.values, color=color(s), linewidth=8.8, label=_disp(s))
    _maybe_shift_marker(sample_shift_marker_x)
    plt.xlabel("Test Sample #")
    plt.ylabel("Balanced Accuracy (cumulative)")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(outdir, "ttt_curve_sample_cumulative")
    plt.close()

    # ---- 3. batch-level BALANCED accuracy (overlay, faint raw + smoothed) ----
    plt.figure(figsize=(11, 7))
    for s in strategies:
        ser = batch_bal_c[s]
        plt.plot(_xvals(ser.index, batch_x_scale), ser.values, color=color(s), alpha=0.20, linewidth=3.2)
        plt.plot(_xvals(ser.index, batch_x_scale), _rolling(ser, max(5, window // 3)).values, color=color(s),
                 linewidth=8.0, label=_disp(s))
    _maybe_shift_marker(shift_marker_x)
    plt.xlabel(batch_xlabel)
    plt.ylabel("Balanced Accuracy")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(outdir, "ttt_curve_batch")
    plt.close()

    # ---- 3b. sliding-window AUC vs sample number ----
    auc_sample = _sliding_auc(df, window=auc_window, stride=auc_stride)
    plt.figure(figsize=(11, 7))
    for s in strategies:
        ser = auc_sample[s]
        plt.plot(ser.index, ser.values, color=color(s), linewidth=8.0, label=_disp(s))
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    _maybe_shift_marker(sample_shift_marker_x)
    plt.xlabel("Test Sample #")
    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(outdir, "ttt_curve_sample_auc")
    plt.close()

    # ---- 3c. per-batch AUC vs batch number (raw + smoothed) ----
    auc_batch = _batch_auc(df)
    plt.figure(figsize=(11, 7))
    for s in strategies:
        ser = auc_batch[s]
        plt.plot(_xvals(ser.index, batch_x_scale), ser.values, color=color(s), alpha=0.20, linewidth=3.2)
        plt.plot(_xvals(ser.index, batch_x_scale), _rolling(ser, max(5, window // 3)).values, color=color(s),
                 linewidth=8.0, label=_disp(s))
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    _maybe_shift_marker(shift_marker_x)
    plt.xlabel(batch_xlabel)
    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(outdir, "ttt_curve_batch_auc")
    plt.close()

    # ---- 3d. per-batch AI recall (TPR @0.5) vs batch number (raw + smoothed) ----
    recall_batch = _batch_recall(df)
    plt.figure(figsize=(11, 7))
    for s in strategies:
        ser = recall_batch[s]
        plt.plot(_xvals(ser.index, batch_x_scale), ser.values, color=color(s), alpha=0.20, linewidth=3.2)
        plt.plot(_xvals(ser.index, batch_x_scale), _rolling(ser, max(5, window // 3)).values, color=color(s),
                 linewidth=8.0, label=_disp(s))
    _maybe_shift_marker(shift_marker_x)
    plt.xlabel(batch_xlabel)
    plt.ylabel("AI Recall")
    plt.ylim(0, 1)
    plt.legend(title="strategy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(outdir, "ttt_curve_batch_recall")
    plt.close()

    # ---- 3e/3f/3g. per-batch mean P(AI|AI), mean P(human|human), human recall ----
    for curvef, ylab, fname in [
        (_batch_meanprob_ai,     "P(AI | AI)",         "ttt_curve_batch_pai_ai"),
        (_batch_meanprob_human,  "P(human | human)",   "ttt_curve_batch_phuman_human"),
        (_batch_human_recall,    "Human Recall",       "ttt_curve_batch_human_recall"),
    ]:
        curves = curvef(df)
        plt.figure(figsize=(11, 7))
        for s in strategies:
            ser = curves.get(s)
            if ser is None or ser.empty:
                continue
            plt.plot(_xvals(ser.index, batch_x_scale), ser.values, color=color(s), alpha=0.20, linewidth=3.2)
            plt.plot(_xvals(ser.index, batch_x_scale), _rolling(ser, max(5, window // 3)).values, color=color(s),
                     linewidth=8.0, label=_disp(s))
        _maybe_shift_marker(shift_marker_x)
        plt.xlabel(batch_xlabel)
        plt.ylabel(ylab)
        plt.ylim(0, 1)
        plt.legend(title="strategy")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        _savefig(outdir, fname)
        plt.close()

    # ---- 4. per-strategy detail (raw + BALANCED moving avg + BALANCED cumulative) ----
    for s in strategies:
        ser = sample_c[s]
        plt.figure(figsize=(11, 7))
        plt.plot(ser.index, ser.values, color=color(s), alpha=0.15, linewidth=3.2,
                 label="per-sample (mean over seeds)")
        plt.plot(bal_sample_c[s].index, bal_sample_c[s].values, color=color(s),
                 linewidth=8.8, label=f"balanced moving avg (w={window})")
        plt.plot(cum_bal_sample_c[s].index, cum_bal_sample_c[s].values, color="black",
                 linewidth=3.0, linestyle="--", label="balanced cumulative")
        _maybe_shift_marker(sample_shift_marker_x)
        plt.xlabel("Test Sample #")
        plt.ylabel("Balanced Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        _savefig(outdir, f"ttt_curve_{s}_sample")
        plt.close()

    # ---- 5. P-set-calibrated BALANCED accuracy (threshold from known-human text, not 0.5) ----
    if "correct_cal" in df.columns:
        _overlay_accuracy(
            _sample_curves(df, "correct_cal"), strategies, color,
            outdir, "ttt_curve_sample_acc_cal",
            "Test Sample #", "Balanced Accuracy (calibrated)", "", window,
            bold_curves=_balanced_rolling(df, "correct_cal", window),
            marker_x=sample_shift_marker_x)
        _overlay_accuracy(
            _batch_balanced_accuracy(df, "correct_cal"), strategies, color,
            outdir, "ttt_curve_batch_acc_cal",
            batch_xlabel, "Balanced Accuracy (calibrated)", "", max(5, window // 3),
            x_scale=batch_x_scale, marker_x=shift_marker_x)

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


def make_ramp_plots(samples_csv, outdir, title_suffix="", batch_size=None, mark_shift=False):
    """Gradual-shift diagnostics (only when ai_source is present): AUC and recall on the
    Z-subset over batches, with the %Z schedule overlaid. The Z-subset curves isolate
    'is the model getting better at rewrite_Z' from 'the stream is getting harder'."""
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(samples_csv)
    if "ai_source" not in df.columns or (df["ai_source"] == "Z").sum() == 0:
        return  # not a gradual run, or no Z present (growth=0)
    strategies = _present(df["strategy"].unique().tolist())
    fracz = _batch_fracz(df)
    batch_x_scale = batch_size if batch_size is not None else 1
    batch_xlabel = "Total # Test-Time Data Seen" if batch_size is not None else "Test Batch #"
    n_batches = int(df["batch_id"].max()) + 1
    shift_marker_x = (n_batches * batch_x_scale / 2.0) if mark_shift and batch_size is not None else None

    for metric, fn, ylab, fname in [
        ("auc", _batch_subset_auc, "AUC on {human vs Z-AI}", "ttt_curve_zsubset_auc"),
        ("recall", _batch_subset_recall, "Z-AI recall", "ttt_curve_zsubset_recall"),
    ]:
        curves = fn(df, "Z")
        plt.figure(figsize=(11, 7))
        for s in strategies:
            ser = curves.get(s)
            if ser is None or ser.empty:
                continue
            w = max(5, len(ser) // 25)
            plt.plot(_xvals(ser.index, batch_x_scale), _rolling(ser, w).values, color=STRATEGY_COLOR.get(s), linewidth=8.8, label=_disp(s))
        plt.plot(_xvals(fracz.index, batch_x_scale), fracz.values, color="black", linestyle=":", linewidth=1.0, label="%Z in batch")
        plt.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
        _maybe_shift_marker(shift_marker_x)
        plt.xlabel(batch_xlabel)
        plt.ylabel(ylab)
        plt.ylim(0, 1)
        plt.legend(title="strategy")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        _savefig(outdir, fname)
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
        if mode in EXCLUDE or mode not in df["strategy"].unique():
            continue
        g = df[df["strategy"] == mode]
        bmin, bmax = int(g["batch_id"].min()), int(g["batch_id"].max())
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
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
            # Panel identifier (first vs last batch) kept as an in-axes annotation, not a title.
            ax.text(0.03, 0.97, name, transform=ax.transAxes, ha="left", va="top", fontsize=20)
            ax.legend(fontsize=18)
        axes[0].set_ylabel("Count")
        plt.tight_layout()
        _savefig(outdir, f"eda_pred_hist_{mode}")
        plt.close()
        print(f"wrote {outdir}/eda_pred_hist_{mode}.png (+.pdf)")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "logging_accuracy_xz_TTT_rewrite_Z_samples.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "figs/ttt"
    make_curves(csv, out)
