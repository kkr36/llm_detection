from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import math
from matplotlib.colors import LinearSegmentedColormap
import os

font = {
    'weight': 'bold',
    'size'  : 35
}
matplotlib.rc('font', **font)

orange_white_purple = LinearSegmentedColormap.from_list(
    "orange_white_purple", ["orange", "white", "purple"][::-1]
)

ci = False  # show confidence interval text in heatmap annotations

def fmt(v):
    s = f"{v:.2f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s

input_file = "../logging_accuracy_xz.csv"
output_folder = input_file.split("/")[-1].split(".csv")[0] + "_paper"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plot_metrics = [
    "auc",
    "accuracy",
    "pos_prob",
    "neg_prob",
    "bce",
    "tpr",
    "bbe",
    "plugin-int",
    # "tnr"
]

name_to_name = {
    "auc"        : "AUC",
    "accuracy"   : "Bal. Accuracy",
    "pos_prob"   : "Avg. P(human | human)",
    "neg_prob"   : "Avg. P(human | AI)",
    # "entropy_pos": "Shannon Entropy Human",
    # "entropy_neg": "Shannon Entropy LLM",
    # "entropy"    : "Avg. Shannon Entropy",
    "bce"        : "Bal. Cross-Entropy",
    "bbe"        : "Bias",
    # "plugin"     : "Bias Plug-In Alpha",
    "plugin-int" : "Bias Avg P(Human)",
    "tpr"        : "Human Recall",
    "tnr"        : "AI Recall"
}

binary_metrics    = ["auc", "accuracy", "pos_prob", "neg_prob", "entropy_pos", "entropy_neg", "entropy", "bce"]
diverging_metrics = ["bbe", "plugin", "plugin-int"]
flip_metrics      = ["pos_prob", "neg_prob", "bbe", "plugin", "plugin-int"]
swap_metrics      = ["entropy_pos", "entropy_neg"]

method_order = ["PN", "TEDn"]
train_llm_order = ["X", "xz", "xzz", "xzzz"]

# Display label for each (method, train_llm) pair
train_llm_label = {
    ("TEDn", "X")    : "PU_0",
    ("TEDn", "xz")   : "PU_1",
    ("TEDn", "xzz")  : "PU_2",
    ("TEDn", "xzzz") : "PU_3",
    ("PN",   "X")    : "PN_0",
    ("PN",   "xz")   : "PN_1",
    ("PN",   "xzz")  : "PN_2",
}

# Eval columns per method type
TEDN_EVAL_COLS = ["rewrite_X", "rewrite_Z", "rewrite_Z_1_PU", "rewrite_Z_2_PU"]
PN_EVAL_COLS   = ["rewrite_X", "rewrite_Z", "rewrite_Z_1_PN", "rewrite_Z_2_PN"]
ALL_EVAL_COLS  = ["rewrite_X", "rewrite_Z", "rewrite_Z_1_PU", "rewrite_Z_2_PU", "rewrite_Z_1_PN", "rewrite_Z_2_PN"]

# Human-readable eval col labels
eval_col_labels = {
    "rewrite_X"     : "X",
    "rewrite_Z"     : "Z",
    "rewrite_Z_1_PU": "Z_1 (PU)",
    "rewrite_Z_2_PU": "Z_2 (PU)",
    "rewrite_Z_1_PN": "Z_1 (PN)",
    "rewrite_Z_2_PN": "Z_2 (PN)",
}

def add_accuracy_cols(df, ci_level=0.95):
    """Balanced accuracy = (TPR + TNR) / 2. In this file's CSV convention,
    pos_prob = TPR (Avg Pred Human) and neg_prob = FPR (Avg Pred LLM)."""
    df = df.copy()
    ci = str(ci_level)
    tpr     = df["tpr"]
    fpr     = df["fpr"]
    tpr_l   = df[f"tpr_l_{ci}"] if f"tpr_l_{ci}" in df.columns else tpr
    tpr_u   = df[f"tpr_u_{ci}"] if f"tpr_u_{ci}" in df.columns else tpr
    fpr_l   = df[f"fpr_l_{ci}"] if f"fpr_l_{ci}" in df.columns else fpr
    fpr_u   = df[f"fpr_u_{ci}"] if f"fpr_u_{ci}" in df.columns else fpr
    df["accuracy"]              = (tpr + 1 - fpr) / 2
    df[f"accuracy_l_{ci}"]     = (tpr_l + 1 - fpr_u) / 2
    df[f"accuracy_u_{ci}"]     = (tpr_u + 1 - fpr_l) / 2
    return df

def resolve_cols(metric, ci_level=0.95):
    """Return (point_col, lower_col, upper_col, do_flip)."""
    pos_bool = "pos" in metric
    has_posneg = "pos" in metric or "neg" in metric

    if metric in swap_metrics:
        base = metric.replace("pos", "neg") if pos_bool else metric.replace("neg", "pos")
        return base, f"{base}_l_{ci_level}", f"{base}_u_{ci_level}", False

    elif metric in flip_metrics:
        base = (metric.replace("pos", "neg") if pos_bool else metric.replace("neg", "pos")) if has_posneg else metric
        return base, f"{base}_l_{ci_level}", f"{base}_u_{ci_level}", True

    else:
        return metric, f"{metric}_l_{ci_level}", f"{metric}_u_{ci_level}", False


# ---------------------------------------------------------------------------
# Plot 1: Heatmap — TEDn rows use TEDN_EVAL_COLS, PN rows use PN_EVAL_COLS
# ---------------------------------------------------------------------------

def make_xz_heatmap(df, metrics, title=False):
    key_cols = ["learning_method", "train_llm", "eval_llm"]
    numeric_cols = [c for c in df.columns if c not in key_cols and df[c].dtype != object]
    df = df.groupby(key_cols)[numeric_cols].mean().reset_index()

    row_keys = [
        (m, l)
        for m in method_order
        for l in train_llm_order
        if not df[(df["learning_method"] == m) & (df["train_llm"] == l)].empty
    ]
    row_labels = [train_llm_label[(m, l)] for m, l in row_keys]

    col_order = ALL_EVAL_COLS
    col_labels = [eval_col_labels[c] for c in col_order]

    for metric in metrics:
        point_col, lower_col, upper_col, do_flip = resolve_cols(metric)

        point_rows, lower_rows, upper_rows = [], [], []

        for m, l in row_keys:
            eval_cols = TEDN_EVAL_COLS if m == "TEDn" else PN_EVAL_COLS
            subset = df[(df["learning_method"] == m) & (df["train_llm"] == l)].set_index("eval_llm")

            def get_val(col, ec, sub=subset):
                if ec not in sub.index or col not in sub.columns:
                    return np.nan
                return sub.loc[ec, col]

            pt_row = [get_val(point_col, ec) if ec in eval_cols else np.nan for ec in col_order]
            lo_row = [get_val(lower_col, ec) if ec in eval_cols else np.nan for ec in col_order]
            hi_row = [get_val(upper_col, ec) if ec in eval_cols else np.nan for ec in col_order]

            point_rows.append(pt_row)
            lower_rows.append(lo_row)
            upper_rows.append(hi_row)

        point_df = pd.DataFrame(point_rows, index=row_labels, columns=col_labels)
        lower_df = pd.DataFrame(lower_rows, index=row_labels, columns=col_labels)
        upper_df = pd.DataFrame(upper_rows, index=row_labels, columns=col_labels)

        plot_df = point_df.copy()
        if metric == "bbe" or metric == "plugin-int":
            plot_df  = plot_df  - 0.5
            lower_df = lower_df - 0.5
            upper_df = upper_df - 0.5

        annot = plot_df.copy().astype(object)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                val = plot_df.iloc[i, j]
                lo  = lower_df.iloc[i, j]
                hi  = upper_df.iloc[i, j]
                if pd.isna(val):
                    annot.iloc[i, j] = ""
                else:
                    annot.iloc[i, j] = (
                        f"{fmt(val)}\n[{fmt(lo)}, {fmt(hi)}]" if ci else fmt(val)
                    )

        n_rows = len(row_labels)
        n_cols = len(col_order)
        plt.figure(figsize=(max(10, n_cols * 6), max(8, n_rows * 3)))

        if metric in binary_metrics:
            cmap   = "YlOrBr"
            center = np.nanmean(plot_df.values.astype(float))
            max_dev = np.nanmax(np.abs(plot_df.values.astype(float) - center))
            vmin, vmax = center - max_dev, center + max_dev
        else:
            cmap    = orange_white_purple
            center  = 0.0 if metric in ["bbe", "plugin-int", "plugin"] else 0.5
            max_dev = np.nanmax(np.abs(plot_df.values.astype(float) - center))
            vmin, vmax = center - max_dev, center + max_dev

        # Mask NaN cells so they render as grey
        mask = plot_df.isna()

        ax = sns.heatmap(
            plot_df,
            annot=annot,
            fmt="",
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            mask=mask,
        )

        # Dividing line between PN and TEDn groups
        boundary = sum(1 for m, _ in row_keys if m == method_order[0])
        ax.axhline(y=boundary, color="black", linewidth=6)

        # Vertical dividers: shared | PU-only | PN-only
        # col_order = [rewrite_X, rewrite_Z, rewrite_Z_1_PU, rewrite_Z_2_PU, rewrite_Z_1_PN, rewrite_Z_2_PN]
        ax.axvline(x=2, color="black", linewidth=3, linestyle="--")
        ax.axvline(x=4, color="black", linewidth=3, linestyle="--")

        if title:
            plt.title(f"{name_to_name.get(metric, metric)} (95% CI)")
        plt.xlabel("Eval Prompt")
        plt.ylabel("Model")

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/xz_heatmap_{metric}.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()


# ---------------------------------------------------------------------------
# Plot 1b: Heatmap — same as above but Z_1_PU and Z_1_PN collapsed into Z_1
#   TEDn rows pull from rewrite_Z_1_PU, PN rows pull from rewrite_Z_1_PN,
#   but both are placed in the same "Z_1" column.
# ---------------------------------------------------------------------------

COLLAPSED_COL_LABELS = ["X", "Z", "Z_1", "Z_2"]

# Which raw eval_llm to read for the Z_1 / Z_2 columns, per method
Z1_SOURCE = {"TEDn": "rewrite_Z_1_PU", "PN": "rewrite_Z_1_PN"}
Z2_SOURCE = {"TEDn": "rewrite_Z_2_PU", "PN": "rewrite_Z_2_PN"}


def make_xz_heatmap_collapsed(df, metrics, title=False):
    """Heatmap with a single Z_1 column (TEDn reads Z_1_PU, PN reads Z_1_PN)."""
    key_cols = ["learning_method", "train_llm", "eval_llm"]
    numeric_cols = [c for c in df.columns if c not in key_cols and df[c].dtype != object]
    df = df.groupby(key_cols)[numeric_cols].mean().reset_index()

    row_keys = [
        (m, l)
        for m in method_order
        for l in train_llm_order
        if not df[(df["learning_method"] == m) & (df["train_llm"] == l)].empty
    ]
    row_labels = [train_llm_label[(m, l)] for m, l in row_keys]

    for metric in metrics:
        point_col, lower_col, upper_col, do_flip = resolve_cols(metric)

        point_rows, lower_rows, upper_rows = [], [], []

        for m, l in row_keys:
            subset = df[(df["learning_method"] == m) & (df["train_llm"] == l)].set_index("eval_llm")
            z1_src = Z1_SOURCE[m]

            def get_val(col, ec, sub=subset):
                if ec not in sub.index or col not in sub.columns:
                    return np.nan
                return sub.loc[ec, col]

            # columns: X, Z, Z_1 (collapsed), Z_2 (collapsed)
            z2_src = Z2_SOURCE[m]
            shared_ecs = ["rewrite_X", "rewrite_Z"]
            pt_row = [get_val(point_col, ec) for ec in shared_ecs] + [get_val(point_col, z1_src), get_val(point_col, z2_src)]
            lo_row = [get_val(lower_col, ec) for ec in shared_ecs] + [get_val(lower_col, z1_src), get_val(lower_col, z2_src)]
            hi_row = [get_val(upper_col, ec) for ec in shared_ecs] + [get_val(upper_col, z1_src), get_val(upper_col, z2_src)]

            point_rows.append(pt_row)
            lower_rows.append(lo_row)
            upper_rows.append(hi_row)

        point_df = pd.DataFrame(point_rows, index=row_labels, columns=COLLAPSED_COL_LABELS)
        lower_df = pd.DataFrame(lower_rows, index=row_labels, columns=COLLAPSED_COL_LABELS)
        upper_df = pd.DataFrame(upper_rows, index=row_labels, columns=COLLAPSED_COL_LABELS)

        plot_df = point_df.copy()
        if metric == "bbe" or metric == "plugin-int":
            plot_df  = plot_df  - 0.5
            lower_df = lower_df - 0.5
            upper_df = upper_df - 0.5

        annot = plot_df.copy().astype(object)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                val = plot_df.iloc[i, j]
                lo  = lower_df.iloc[i, j]
                hi  = upper_df.iloc[i, j]
                if pd.isna(val):
                    annot.iloc[i, j] = ""
                else:
                    annot.iloc[i, j] = (
                        f"{fmt(val)}\n[{fmt(lo)}, {fmt(hi)}]" if ci else fmt(val)
                    )

        n_rows = len(row_labels)
        n_cols = len(COLLAPSED_COL_LABELS)
        plt.figure(figsize=(max(10, n_cols * 6), max(8, n_rows * 3)))

        if metric in binary_metrics:
            cmap    = "YlOrBr"
            center  = np.nanmean(plot_df.values.astype(float))
            max_dev = np.nanmax(np.abs(plot_df.values.astype(float) - center))
            vmin, vmax = center - max_dev, center + max_dev
        else:
            cmap    = orange_white_purple
            center  = 0.0 if metric in ["bbe", "plugin-int", "plugin"] else 0.5
            max_dev = np.nanmax(np.abs(plot_df.values.astype(float) - center))
            vmin, vmax = center - max_dev, center + max_dev

        ax = sns.heatmap(
            plot_df,
            annot=annot,
            fmt="",
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
        )

        # Dividing line between PN and TEDn groups
        boundary = sum(1 for mm, _ in row_keys if mm == method_order[0])
        ax.axhline(y=boundary, color="black", linewidth=6)

        if title:
            plt.title(f"{name_to_name.get(metric, metric)} (95% CI)")
        plt.xlabel("Eval Prompt")
        plt.ylabel("Model")

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/xz_heatmap_collapsed_{metric}.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()


# ---------------------------------------------------------------------------
# Plot 2: Line plot — PU trajectory vs PN trajectory across timesteps
# ---------------------------------------------------------------------------
#
# PU (TEDn) line:
#   t=0 : learning_method=TEDn, train_llm=X,   eval_llm=rewrite_X
#   t=1 : learning_method=TEDn, train_llm=xz,  eval_llm=rewrite_Z
#   t=2 : learning_method=TEDn, train_llm=xzz, eval_llm=rewrite_Z_1_PU
#
# PN line:
#   t=0 : learning_method=PN,   train_llm=X,   eval_llm=rewrite_X
#   t=1 : learning_method=PN,   train_llm=X,   eval_llm=rewrite_Z
#   t=2 : learning_method=PN,   train_llm=xz,  eval_llm=rewrite_Z_1_PN

PU_TRAJECTORY = [
    ("TEDn", "X",     "rewrite_X"),
    ("TEDn", "xz",    "rewrite_Z"),
    ("TEDn", "xzz",   "rewrite_Z_1_PU"),
    ("TEDn", "xzzz",  "rewrite_Z_2_PU"),
]

PN_TRAJECTORY = [
    ("PN", "X",   "rewrite_X"),
    ("PN", "X",   "rewrite_Z"),
    ("PN", "xz",  "rewrite_Z_1_PN"),
    ("PN", "xzz", "rewrite_Z_2_PN"),
]

TIMESTEP_LABELS = ["t=0", "t=1", "t=2", "t=3"]


def _get_trajectory_vals(df, trajectory, point_col, lower_col, upper_col):
    """Return arrays of (point, lower, upper) for each step in trajectory."""
    pts, los, his = [], [], []
    for method, train, evl in trajectory:
        row = df[
            (df["learning_method"] == method) &
            (df["train_llm"] == train) &
            (df["eval_llm"] == evl)
        ]
        if row.empty or point_col not in row.columns:
            pts.append(np.nan); los.append(np.nan); his.append(np.nan)
        else:
            pts.append(row[point_col].mean())
            los.append(row[lower_col].mean() if lower_col in row.columns else np.nan)
            his.append(row[upper_col].mean() if upper_col in row.columns else np.nan)
    return np.array(pts), np.array(los), np.array(his)


def make_xz_lineplot(df, metrics, title=False):
    timesteps = np.arange(len(TIMESTEP_LABELS))

    for metric in metrics:
        point_col, lower_col, upper_col, do_flip = resolve_cols(metric)

        pu_pts, pu_los, pu_his = _get_trajectory_vals(df, PU_TRAJECTORY, point_col, lower_col, upper_col)
        pn_pts, pn_los, pn_his = _get_trajectory_vals(df, PN_TRAJECTORY, point_col, lower_col, upper_col)

        if metric == "bbe" or metric == "plugin-int":
            pu_pts -= 0.5; pu_los -= 0.5; pu_his -= 0.5
            pn_pts -= 0.5; pn_los -= 0.5; pn_his -= 0.5

        fig, ax = plt.subplots(figsize=(12, 7))

        # PU (TEDn) line
        ax.plot(timesteps, pu_pts, marker="o", linewidth=3, markersize=10,
                label="PU + TTA", color="purple")
        ax.fill_between(timesteps, pu_los, pu_his, alpha=0.2, color="purple")

        # PN line
        ax.plot(timesteps, pn_pts, marker="s", linewidth=3, markersize=10,
                label="Supervised", color="orange")
        ax.fill_between(timesteps, pn_los, pn_his, alpha=0.2, color="orange")

        ax.set_xticks(timesteps)
        ax.set_xticklabels(TIMESTEP_LABELS, fontsize=22)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(name_to_name.get(metric, metric))
        if title:
            ax.set_title(f"{name_to_name.get(metric, metric)} — PU vs PN trajectory")
        ax.legend(fontsize=24)

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/xz_lineplot_{metric}.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Pangram helpers (used by bar plots with Pangram comparison)
# ---------------------------------------------------------------------------

PANGRAM_SCORE_TYPE = "dominant_category"
PANGRAM_COLOR      = "steelblue"


def _get_pangram_vals(df, eval_llm_col, point_col, lower_col, upper_col,
                      score_type=PANGRAM_SCORE_TYPE):
    """Return (point, lower, upper) for pangram at a given eval_llm."""
    row = df[
        (df["learning_method"] == "pangram") &
        (df["eval_llm"] == eval_llm_col) &
        (df["pangram_score_type"] == score_type)
    ]
    if row.empty or point_col not in row.columns:
        return np.nan, np.nan, np.nan
    pt = row[point_col].mean()
    lo = row[lower_col].mean() if lower_col in row.columns else np.nan
    hi = row[upper_col].mean() if upper_col in row.columns else np.nan
    if "pos_prob" in point_col or "neg_prob" in point_col:
        pt, lo, hi = 1 - pt, 1 - hi, 1 - lo  # lo/hi swap since 1-x reverses order
    if "plugin-int" in point_col:
        pt, lo, hi = 1 - pt, 1 - hi, 1 - lo  # lo/hi swap since 1-x reverses order
    return pt, lo, hi


def _safe_yerr(pts, los, his):
    """Build a (2, N) yerr array; clip negative half-widths to 0."""
    lo_err = np.where(np.isnan(pts - los), 0, np.clip(pts - los, 0, None))
    hi_err = np.where(np.isnan(his - pts), 0, np.clip(his - pts, 0, None))
    return np.array([lo_err, hi_err])


# ---------------------------------------------------------------------------
# Plot 2b: Bar plot — same data as lineplot but grouped bars by model
# ---------------------------------------------------------------------------

def make_xz_barplot(df, metrics, title=False):
    """Bar chart version of make_xz_lineplot: grouped bars (PU, PN) per timestep."""
    n_groups  = len(TIMESTEP_LABELS)
    bar_width = 0.35
    x = np.arange(n_groups)

    for metric in metrics:
        point_col, lower_col, upper_col, _ = resolve_cols(metric)

        pu_pts, pu_los, pu_his = _get_trajectory_vals(
            df, PU_TRAJECTORY, point_col, lower_col, upper_col)
        pn_pts, pn_los, pn_his = _get_trajectory_vals(
            df, PN_TRAJECTORY, point_col, lower_col, upper_col)

        if metric in ("bbe", "plugin-int"):
            pu_pts -= 0.5; pu_los -= 0.5; pu_his -= 0.5
            pn_pts -= 0.5; pn_los -= 0.5; pn_his -= 0.5

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.bar(x - bar_width / 2, pu_pts, bar_width, label="PU + TTA", color="purple",
               yerr=_safe_yerr(pu_pts, pu_los, pu_his), capsize=5,
               error_kw={"linewidth": 2})
        ax.bar(x + bar_width / 2, pn_pts, bar_width, label="Supervised", color="orange",
               yerr=_safe_yerr(pn_pts, pn_los, pn_his), capsize=5,
               error_kw={"linewidth": 2})

        ax.set_xticks(x)
        ax.set_xticklabels(TIMESTEP_LABELS, fontsize=22)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(name_to_name.get(metric, metric))
        if title:
            ax.set_title(f"{name_to_name.get(metric, metric)} — PU vs PN trajectory")
        ax.legend(fontsize=24)

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/xz_barplot_{metric}.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3a (Fig A): t=0 and t=1 only — PU, PN, Pangram grouped bars
# ---------------------------------------------------------------------------

def make_xz_barplot_fig_a(df, metrics, title=False):
    """
    Fig A: grouped bars at t=0 and t=1 for PU, PN, and Pangram.
      Pangram t=0 → eval_llm=rewrite_X
      Pangram t=1 → eval_llm=rewrite_Z
    """
    ts_labels = ["Naive prompt", "Adversarial humanizing prompt"]
    bar_width  = 0.25
    x = np.arange(len(ts_labels))
    offsets = [-bar_width, 0.0, bar_width]

    for metric in metrics:
        point_col, lower_col, upper_col, _ = resolve_cols(metric)

        pu_pts, pu_los, pu_his = _get_trajectory_vals(
            df, PU_TRAJECTORY[:2], point_col, lower_col, upper_col)
        pn_pts, pn_los, pn_his = _get_trajectory_vals(
            df, PN_TRAJECTORY[:2], point_col, lower_col, upper_col)

        pg_t0 = _get_pangram_vals(df, "rewrite_X", point_col, lower_col, upper_col)
        pg_t1 = _get_pangram_vals(df, "rewrite_Z", point_col, lower_col, upper_col)
        pg_pts = np.array([pg_t0[0], pg_t1[0]])
        pg_los = np.array([pg_t0[1], pg_t1[1]])
        pg_his = np.array([pg_t0[2], pg_t1[2]])

        if metric in ("bbe", "plugin-int"):
            pu_pts -= 0.5; pu_los -= 0.5; pu_his -= 0.5
            pn_pts -= 0.5; pn_los -= 0.5; pn_his -= 0.5
            pg_pts -= 0.5; pg_los -= 0.5; pg_his -= 0.5

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.bar(x + offsets[0], pu_pts, bar_width, label="PU + TTA", color="purple",
               yerr=_safe_yerr(pu_pts, pu_los, pu_his), capsize=5,
               error_kw={"linewidth": 2})
        ax.bar(x + offsets[1], pn_pts, bar_width, label="Supervised", color="orange",
               yerr=_safe_yerr(pn_pts, pn_los, pn_his), capsize=5,
               error_kw={"linewidth": 2})
        ax.bar(x + offsets[2], pg_pts, bar_width, label="Pangram", color=PANGRAM_COLOR,
               yerr=_safe_yerr(pg_pts, pg_los, pg_his), capsize=5,
               error_kw={"linewidth": 2})

        ax.set_xticks(x)
        ax.set_xticklabels(ts_labels, fontsize=22)
        ax.set_ylabel(name_to_name.get(metric, metric))
        if title:
            ax.set_title(f"{name_to_name.get(metric, metric)} — PU, PN, Pangram (t=0,1)")

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/xz_barplot_fig_a_{metric}.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3b (Fig B): t=1, t=2, t=3 — PU and PN only (no Pangram)
# ---------------------------------------------------------------------------

def make_xz_barplot_fig_b(df, metrics, title=False):
    """
    Fig B: grouped bars at Iterations 1-3 plus Average for PU and PN (no Pangram).
    """
    iter_labels = ["Iteration 1", "Iteration 2", "Iteration 3", "Average"]
    bar_width   = 0.35
    x = np.arange(len(iter_labels))

    for metric in metrics:
        point_col, lower_col, upper_col, _ = resolve_cols(metric)

        pu_pts, pu_los, pu_his = _get_trajectory_vals(
            df, PU_TRAJECTORY[1:], point_col, lower_col, upper_col)
        pn_pts, pn_los, pn_his = _get_trajectory_vals(
            df, PN_TRAJECTORY[1:], point_col, lower_col, upper_col)

        if metric in ("bbe", "plugin-int"):
            pu_pts -= 0.5; pu_los -= 0.5; pu_his -= 0.5
            pn_pts -= 0.5; pn_los -= 0.5; pn_his -= 0.5

        # Append average bar values
        pu_avg_pt = np.array([np.nanmean(pu_pts)])
        pn_avg_pt = np.array([np.nanmean(pn_pts)])
        pu_avg_lo = np.array([np.nanmean(pu_los)])
        pn_avg_lo = np.array([np.nanmean(pn_los)])
        pu_avg_hi = np.array([np.nanmean(pu_his)])
        pn_avg_hi = np.array([np.nanmean(pn_his)])

        pu_all_pts = np.append(pu_pts, pu_avg_pt)
        pn_all_pts = np.append(pn_pts, pn_avg_pt)
        pu_all_los = np.append(pu_los, pu_avg_lo)
        pn_all_los = np.append(pn_los, pn_avg_lo)
        pu_all_his = np.append(pu_his, pu_avg_hi)
        pn_all_his = np.append(pn_his, pn_avg_hi)

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.bar(x - bar_width / 2, pu_all_pts, bar_width, label="PU + TTA", color="purple",
               yerr=_safe_yerr(pu_all_pts, pu_all_los, pu_all_his), capsize=5,
               error_kw={"linewidth": 2})
        ax.bar(x + bar_width / 2, pn_all_pts, bar_width, label="Supervised", color="orange",
               yerr=_safe_yerr(pn_all_pts, pn_all_los, pn_all_his), capsize=5,
               error_kw={"linewidth": 2})

        # Vertical separator before the Average bar
        ax.axvline(x=len(iter_labels) - 1 - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(iter_labels, fontsize=22)
        ax.set_xlabel("Adversarial iterations")
        ax.set_ylabel(name_to_name.get(metric, metric))
        if title:
            ax.set_title(f"{name_to_name.get(metric, metric)} — PU vs PN (adversarial iterations)")

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/xz_barplot_fig_b_{metric}.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3c (Fig AB): Fig A and Fig B as side-by-side subfigures, shared legend
# ---------------------------------------------------------------------------

def make_xz_barplot_fig_ab(df, metrics, title=False):
    """
    Combined figure: Fig A (left, t=0 and t=1 with Pangram) and Fig B (right,
    adversarial iterations) as subfigures, with a shared legend above both panels.
    """
    import matplotlib.patches as mpatches

    ts_labels   = ["Naive\nprompt", "Adversarial\nhumanizing\nprompt"]
    iter_labels = ["Iteration\n1", "Iteration\n2", "Iteration\n3"]

    for metric in metrics:
        point_col, lower_col, upper_col, _ = resolve_cols(metric)

        # --- Fig A data ---
        pu_a_pts, pu_a_los, pu_a_his = _get_trajectory_vals(
            df, PU_TRAJECTORY[:2], point_col, lower_col, upper_col)
        pn_a_pts, pn_a_los, pn_a_his = _get_trajectory_vals(
            df, PN_TRAJECTORY[:2], point_col, lower_col, upper_col)

        pg_t0 = _get_pangram_vals(df, "rewrite_X", point_col, lower_col, upper_col)
        pg_t1 = _get_pangram_vals(df, "rewrite_Z", point_col, lower_col, upper_col)
        pg_pts = np.array([pg_t0[0], pg_t1[0]])
        pg_los = np.array([pg_t0[1], pg_t1[1]])
        pg_his = np.array([pg_t0[2], pg_t1[2]])

        # --- Fig B data ---
        pu_b_pts, pu_b_los, pu_b_his = _get_trajectory_vals(
            df, PU_TRAJECTORY[1:], point_col, lower_col, upper_col)
        pn_b_pts, pn_b_los, pn_b_his = _get_trajectory_vals(
            df, PN_TRAJECTORY[1:], point_col, lower_col, upper_col)

        if metric in ("bbe", "plugin-int"):
            pu_a_pts -= 0.5; pu_a_los -= 0.5; pu_a_his -= 0.5
            pn_a_pts -= 0.5; pn_a_los -= 0.5; pn_a_his -= 0.5
            pg_pts   -= 0.5; pg_los   -= 0.5; pg_his   -= 0.5
            pu_b_pts -= 0.5; pu_b_los -= 0.5; pu_b_his -= 0.5
            pn_b_pts -= 0.5; pn_b_los -= 0.5; pn_b_his -= 0.5

        pu_b_avg = np.nanmean(pu_b_pts)
        pn_b_avg = np.nanmean(pn_b_pts)

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7))

        # --- Draw Fig A ---
        bw_a = 0.25
        x_a  = np.arange(len(ts_labels))
        offs = [-bw_a, 0.0, bw_a]

        ax_a.bar(x_a + offs[0], pu_a_pts, bw_a, color="purple",
                 yerr=_safe_yerr(pu_a_pts, pu_a_los, pu_a_his), capsize=5,
                 error_kw={"linewidth": 2})
        ax_a.bar(x_a + offs[1], pn_a_pts, bw_a, color="orange",
                 yerr=_safe_yerr(pn_a_pts, pn_a_los, pn_a_his), capsize=5,
                 error_kw={"linewidth": 2})
        ax_a.bar(x_a + offs[2], pg_pts, bw_a, color=PANGRAM_COLOR,
                 yerr=_safe_yerr(pg_pts, pg_los, pg_his), capsize=5,
                 error_kw={"linewidth": 2})

        ax_a.set_xticks(x_a)
        ax_a.set_xticklabels(ts_labels, fontsize=22)
        ax_a.set_xlabel("Single-shot adversarial prompt", fontsize=25)
        ax_a.set_ylabel(name_to_name.get(metric, metric))

        # --- Draw Fig B ---
        bw_b = 0.35
        x_b  = np.arange(len(iter_labels))

        ax_b.bar(x_b - bw_b / 2, pu_b_pts, bw_b, color="purple",
                 yerr=_safe_yerr(pu_b_pts, pu_b_los, pu_b_his), capsize=5,
                 error_kw={"linewidth": 2})
        ax_b.bar(x_b + bw_b / 2, pn_b_pts, bw_b, color="orange",
                 yerr=_safe_yerr(pn_b_pts, pn_b_los, pn_b_his), capsize=5,
                 error_kw={"linewidth": 2})

        ax_b.axhline(y=pu_b_avg, color="purple", linewidth=2.5, linestyle="--", alpha=0.8)
        ax_b.axhline(y=pn_b_avg, color="orange", linewidth=2.5, linestyle="--", alpha=0.8)
        ax_b.set_xticks(x_b)
        ax_b.set_xticklabels(iter_labels, fontsize=22)
        ax_b.set_xlabel("Iterated detector-evader game", fontsize=25, labelpad=30)
        # ax_b.set_ylabel(name_to_name.get(metric, metric))

        # --- Shared legend above both panels ---
        legend_handles = [
            mpatches.Patch(color="purple",     label="PU + TTA"),
            mpatches.Patch(color="orange",     label="Supervised"),
            mpatches.Patch(color=PANGRAM_COLOR, label="Pangram"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=24,
                   bbox_to_anchor=(0.5, 1.06), frameon=False)

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/xz_barplot_fig_ab_{metric}.pdf",
            format="pdf",
            bbox_inches="tight"
        )
        plt.clf()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Legend figure — standalone image to place beneath fig a / fig b
# ---------------------------------------------------------------------------

def make_legend_figure():
    """Produce a standalone legend image for fig a and fig b."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(12, 1.2))
    handles = [
        mpatches.Patch(color="purple",      label="PU + TTA"),
        mpatches.Patch(color="orange",      label="Supervised"),
        mpatches.Patch(color=PANGRAM_COLOR, label="Pangram"),
    ]
    ax.legend(handles=handles, fontsize=24, loc="center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 0.5))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"{output_folder}/legend_fig_ab.pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.clf()
    plt.close(fig)


def swap_pangram_tpr_tnr(df, ci_level=0.95):
    """Swap tpr/tnr (and their CIs) for pangram rows only."""
    df = df.copy()
    ci = str(ci_level)
    mask = df["learning_method"] == "pangram"

    tpr_cols = ["tpr", f"tpr_l_{ci}", f"tpr_u_{ci}"]
    tnr_cols = ["tnr", f"tnr_l_{ci}", f"tnr_u_{ci}"]

    for t, n in zip(tpr_cols, tnr_cols):
        if t in df.columns and n in df.columns:
            df.loc[mask, [t, n]] = df.loc[mask, [n, t]].values

    return df


def _make_xz_barplot_fig_ab_grid_part(df, metrics, filename_suffix, title=True):
    """Render one grid PDF for the given subset of metrics."""
    import matplotlib.patches as mpatches

    ts_labels   = ["Naive\nprompt", "Adversarial\nhumanizing\nprompt"]
    iter_labels = ["Iteration\n1", "Iteration\n2", "Iteration\n3"]
    n = len(metrics)

    with matplotlib.rc_context({"font.size": 14, "font.weight": "bold"}):
        fig, axes = plt.subplots(n, 2, figsize=(18, 7 * n), squeeze=False)

        for row_i, metric in enumerate(metrics):
            point_col, lower_col, upper_col, _ = resolve_cols(metric)

            pu_a_pts, pu_a_los, pu_a_his = _get_trajectory_vals(df, PU_TRAJECTORY[:2], point_col, lower_col, upper_col)
            pn_a_pts, pn_a_los, pn_a_his = _get_trajectory_vals(df, PN_TRAJECTORY[:2], point_col, lower_col, upper_col)
            pg_t0 = _get_pangram_vals(df, "rewrite_X", point_col, lower_col, upper_col)
            pg_t1 = _get_pangram_vals(df, "rewrite_Z", point_col, lower_col, upper_col)
            pg_pts = np.array([pg_t0[0], pg_t1[0]])
            pg_los = np.array([pg_t0[1], pg_t1[1]])
            pg_his = np.array([pg_t0[2], pg_t1[2]])

            pu_b_pts, pu_b_los, pu_b_his = _get_trajectory_vals(df, PU_TRAJECTORY[1:], point_col, lower_col, upper_col)
            pn_b_pts, pn_b_los, pn_b_his = _get_trajectory_vals(df, PN_TRAJECTORY[1:], point_col, lower_col, upper_col)

            if metric in ("bbe", "plugin-int"):
                pu_a_pts -= 0.5; pu_a_los -= 0.5; pu_a_his -= 0.5
                pn_a_pts -= 0.5; pn_a_los -= 0.5; pn_a_his -= 0.5
                pg_pts   -= 0.5; pg_los   -= 0.5; pg_his   -= 0.5
                pu_b_pts -= 0.5; pu_b_los -= 0.5; pu_b_his -= 0.5
                pn_b_pts -= 0.5; pn_b_los -= 0.5; pn_b_his -= 0.5

            pu_b_avg = np.nanmean(pu_b_pts)
            pn_b_avg = np.nanmean(pn_b_pts)

            ax_a = axes[row_i, 0]
            ax_b = axes[row_i, 1]

            bw_a = 0.25
            x_a  = np.arange(len(ts_labels))
            offs = [-bw_a, 0.0, bw_a]
            ax_a.bar(x_a + offs[0], pu_a_pts, bw_a, color="purple",
                     yerr=_safe_yerr(pu_a_pts, pu_a_los, pu_a_his), capsize=4, error_kw={"linewidth": 1.5})
            ax_a.bar(x_a + offs[1], pn_a_pts, bw_a, color="orange",
                     yerr=_safe_yerr(pn_a_pts, pn_a_los, pn_a_his), capsize=4, error_kw={"linewidth": 1.5})
            ax_a.bar(x_a + offs[2], pg_pts, bw_a, color=PANGRAM_COLOR,
                     yerr=_safe_yerr(pg_pts, pg_los, pg_his), capsize=4, error_kw={"linewidth": 1.5})
            ax_a.set_xticks(x_a)
            ax_a.set_xticklabels(ts_labels, fontsize=20)
            ax_a.set_xlabel("Single-shot adversarial prompt", fontsize=20)
            ax_a.set_ylabel(name_to_name.get(metric, metric), fontsize=20)
            ax_a.set_title(name_to_name.get(metric, metric), fontsize=20, fontweight="bold")

            bw_b = 0.35
            x_b  = np.arange(len(iter_labels))
            ax_b.bar(x_b - bw_b / 2, pu_b_pts, bw_b, color="purple",
                     yerr=_safe_yerr(pu_b_pts, pu_b_los, pu_b_his), capsize=4, error_kw={"linewidth": 1.5})
            ax_b.bar(x_b + bw_b / 2, pn_b_pts, bw_b, color="orange",
                     yerr=_safe_yerr(pn_b_pts, pn_b_los, pn_b_his), capsize=4, error_kw={"linewidth": 1.5})
            ax_b.axhline(y=pu_b_avg, color="purple", linewidth=2, linestyle="--", alpha=0.8)
            ax_b.axhline(y=pn_b_avg, color="orange", linewidth=2, linestyle="--", alpha=0.8)
            ax_b.set_xticks(x_b)
            ax_b.set_xticklabels(iter_labels, fontsize=20)
            ax_b.set_xlabel("Iterated detector-evader game", fontsize=20, labelpad=25)
            # if metric == "bbe": import pdb; pdb.set_trace()
            ax_b.set_title(name_to_name.get(metric, metric), fontsize=20, fontweight="bold")

        legend_handles = [
            mpatches.Patch(color="purple",      label="PU + TTA"),
            mpatches.Patch(color="orange",       label="Supervised"),
            mpatches.Patch(color=PANGRAM_COLOR,  label="Pangram"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=20,
                   bbox_to_anchor=(0.5, 1.01), frameon=False)

        save_folder = f"{output_folder}/titled" if title else output_folder
        os.makedirs(save_folder, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            f"{save_folder}/xz_barplot_fig_ab_grid_{filename_suffix}.pdf",
            format="pdf", bbox_inches="tight"
        )
        plt.clf()
        plt.close(fig)


def make_xz_barplot_fig_ab_grid(df, metrics, title=True):
    """Two PDFs: first half of metrics → _1.pdf, second half → _2.pdf."""
    segments = 2
    per_segment = len(metrics) // segments
    leftover = 0 if len(metrics) % segments == 0 else 1
    metrics_segmented = [metrics[per_segment*i:per_segment*(i+1)] for i in range(segments+leftover)]
    import pdb; pdb.set_trace()
    for i, segment in enumerate(metrics_segmented):
        # if i != 2: continue
        _make_xz_barplot_fig_ab_grid_part(df, segment,  str(i), title=title)
        # _make_xz_barplot_fig_ab_grid_part(df, metrics[mid:],  "2", title=title)


if __name__ == "__main__":
    data = pd.read_csv(input_file)
    data = swap_pangram_tpr_tnr(data)
    data = add_accuracy_cols(data)
    for use_title in [False, True][1:]:
        # make_xz_heatmap(data, plot_metrics, title=use_title)
        # make_xz_heatmap_collapsed(data, plot_metrics, title=use_title)
        # make_xz_lineplot(data, plot_metrics, title=use_title)
        # make_xz_barplot(data, plot_metrics, title=use_title)
        # make_xz_barplot_fig_a(data, plot_metrics, title=use_title)
        # make_xz_barplot_fig_b(data, plot_metrics, title=use_title)
        # make_xz_barplot_fig_ab(data, plot_metrics, title=use_title)
        make_xz_barplot_fig_ab_grid(data, plot_metrics, title=use_title)

    # make_legend_figure()
