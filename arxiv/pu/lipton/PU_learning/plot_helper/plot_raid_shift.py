"""
Plots for logging_accuracy_raid_shift.csv.

- shift_col in {repetition_penalty, decoding}:
    Bar plot with PN on the left and PU on the right.

- shift_col == domain:
    Two side-by-side heatmaps (left = PU, right = PN),
    y-axis = source_val, x-axis = target_val.

- shift_col == model:
    Single heatmap: PN block on top (y = source_val, x = target_val)
    with PU as a single appended row at the bottom (source_val = "none").
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rc("font", weight="bold", size=22)

INPUT_CSV = "../logging_accuracy_raid_shift_seed_5.csv"
OUTPUT_FOLDER = "logging_accuracy_raid_shift_5"

METRIC = "auc"
CI = "0.95"

METHOD_DISPLAY = {"PN": "Supervised (PN)", "TEDn": "PU + TTA"}
METHOD_ORDER = ["PN", "TEDn"]
COLORS = {"PN": "steelblue", "TEDn": "darkorange"}

orange_white_purple = LinearSegmentedColormap.from_list(
    "orange_white_purple", ["orange", "white", "purple"][::-1]
)

SHIFT_DISPLAY = {
    "repetition_penalty": "Repetition Penalty",
    "decoding": "Decoding Strategy",
}

SHIFT_VAL_DISPLAY = {
    "no": "No Penalty",
    "yes": "With Penalty",
    "greedy": "Greedy",
    "sampling": "Sampling",
}

MODEL_PLOT_METRICS = ["auc", "accuracy", "pos_prob", "neg_prob", "bce", "tpr", "bbe", "plugin-int"]

BINARY_METRICS = {"auc", "accuracy", "pos_prob", "neg_prob", "bce", "tpr"}
DIVERGING_METRICS = {"bbe", "plugin-int"}
REVERSED_METRICS = {"bce", "neg_prob"}

NAME_TO_NAME = {
    "auc": "AUC",
    "accuracy": "Bal. Accuracy",
    "pos_prob": "Avg. P(human | human)",
    "neg_prob": "Avg. P(human | AI)",
    "bce": "Bal. Cross-Entropy",
    "bbe": "Bias",
    "plugin-int": "Bias Avg P(AI)",
    "tpr": "Human Recall",
}


def add_accuracy_cols(df, ci_level=0.95):
    df = df.copy()
    ci_s = str(ci_level)
    tpr, fpr = df["tpr"], df["fpr"]
    df["accuracy"] = (tpr + 1 - fpr) / 2
    df[f"accuracy_l_{ci_s}"] = (df[f"tpr_l_{ci_s}"] + 1 - df[f"fpr_u_{ci_s}"]) / 2
    df[f"accuracy_u_{ci_s}"] = (df[f"tpr_u_{ci_s}"] + 1 - df[f"fpr_l_{ci_s}"]) / 2
    return df


def reverse_bias(df, ci_level=0.95):
    df = df.copy()
    ci_s = str(ci_level)
    df["bbe"] = 1 - df["bbe"]
    df[f"bbe_l_{ci_s}"] = 1 - df[f"bbe_l_{ci_s}"]
    df[f"bbe_u_{ci_s}"] = 1 - df[f"bbe_u_{ci_s}"]
    return df


def reverse_plugin(df, ci_level=0.95):
    df = df.copy()
    ci_s = str(ci_level)
    df["plugin-int"] = 1 - df["plugin-int"]
    df[f"plugin-int_l_{ci_s}"] = 1 - df[f"plugin-int_l_{ci_s}"]
    df[f"plugin-int_u_{ci_s}"] = 1 - df[f"plugin-int_u_{ci_s}"]
    return df


def fmt(v):
    if pd.isna(v) or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = f"{v:.2f}"
    if s.startswith("0."):
        return s[1:]
    elif s.startswith("-0."):
        return "-" + s[2:]
    return s


def _deduplicate(df):
    """Keep the row with the highest run_id per (shift_col, source_val, target_val, train_method)."""
    key = ["shift_col", "source_val", "target_val", "train_method"]
    return (
        df.sort_values("run_id")
        .drop_duplicates(subset=key, keep="last")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Bar plot for repetition_penalty / decoding
# ---------------------------------------------------------------------------

def plot_bar_shifts(df):
    """
    One grouped bar chart combining all shifts with shift_col in
    {repetition_penalty, decoding}. Each group = one shift type,
    bars within group = PN (left) and PU (right).
    """
    shifts = ["repetition_penalty", "decoding"]
    sub = df[df["shift_col"].isin(shifts)]

    n_shifts = len(shifts)
    n_methods = len(METHOD_ORDER)
    bar_width = 0.30
    group_gap = 0.15
    group_width = n_methods * bar_width + group_gap
    x_centers = np.arange(n_shifts) * group_width

    fig, ax = plt.subplots(figsize=(8, 5))

    for m_idx, method in enumerate(METHOD_ORDER):
        heights, errs_lo, errs_hi = [], [], []
        for shift in shifts:
            row = sub[(sub["shift_col"] == shift) & (sub["train_method"] == method)]
            if len(row) == 0:
                heights.append(0.0)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
            else:
                row = row.iloc[0]
                val = row[METRIC]
                lo = val - row[f"{METRIC}_l_{CI}"]
                hi = row[f"{METRIC}_u_{CI}"] - val
                heights.append(val)
                errs_lo.append(lo)
                errs_hi.append(hi)

        offset = (m_idx - (n_methods - 1) / 2) * bar_width
        x_pos = x_centers + offset
        ax.bar(x_pos, heights, width=bar_width,
               label=METHOD_DISPLAY[method], color=COLORS[method], alpha=0.85, zorder=3)
        ax.errorbar(x_pos, heights, yerr=[errs_lo, errs_hi],
                    fmt="none", ecolor="black", elinewidth=1.5, capsize=5, zorder=4)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([SHIFT_DISPLAY[s] for s in shifts], fontsize=18, fontweight="bold")
    ax.set_ylabel("AUC", fontsize=20, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=15, frameon=True, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    save_path = os.path.join(OUTPUT_FOLDER, "bar_shift_rp_decoding.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Side-by-side heatmaps for domain shift
# ---------------------------------------------------------------------------

def _build_domain_pivot(sub, method, metric=METRIC):
    source_vals = sorted(sub["source_val"].unique())
    target_vals = sorted(sub["target_val"].unique())
    m = sub[sub["train_method"] == method]
    point = (
        m.pivot_table(index="source_val", columns="target_val", values=metric, aggfunc="first")
        .reindex(index=source_vals, columns=target_vals)
    )
    lower = (
        m.pivot_table(index="source_val", columns="target_val",
                      values=f"{metric}_l_{CI}", aggfunc="first")
        .reindex(index=source_vals, columns=target_vals)
    )
    upper = (
        m.pivot_table(index="source_val", columns="target_val",
                      values=f"{metric}_u_{CI}", aggfunc="first")
        .reindex(index=source_vals, columns=target_vals)
    )
    return point, lower, upper


def plot_domain_heatmaps(df):
    sub = df[df["shift_col"] == "domain"]
    source_vals = sorted(sub["source_val"].unique())
    target_vals = sorted(sub["target_val"].unique())
    n_rows = len(source_vals)
    n_cols = len(target_vals)

    fig, axes = plt.subplots(1, 2, figsize=(7 * n_cols, 4 * n_rows + 2),
                             gridspec_kw={"wspace": 0.35})

    panel_info = [
        ("TEDn", "PU + TTA", axes[0]),
        ("PN",   "Supervised (PN)", axes[1]),
    ]

    all_vals = []
    for method, _, _ in panel_info:
        p, _, _ = _build_domain_pivot(sub, method)
        all_vals.extend(p.values.flatten().tolist())
    all_vals = [v for v in all_vals if not (isinstance(v, float) and math.isnan(v))]
    global_min = min(all_vals)
    global_max = max(all_vals)
    margin = max((global_max - global_min) * 0.05, 0.01)
    vmin = max(0.0, global_min - margin)
    vmax = min(1.0, global_max + margin)
    center = (vmin + vmax) / 2

    for method, panel_title, ax in panel_info:
        point, lower, upper = _build_domain_pivot(sub, method)

        annot = point.copy().astype(object)
        for i in range(point.shape[0]):
            for j in range(point.shape[1]):
                val = point.iloc[i, j]
                annot.iloc[i, j] = "" if pd.isna(val) else fmt(val)

        sns.heatmap(point, annot=annot, fmt="", cmap="YlOrBr",
                    center=center, vmin=vmin, vmax=vmax, ax=ax,
                    annot_kws={"size": 20})
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: fmt(x))
        )
        ax.set_title(panel_title, fontsize=24, fontweight="bold")
        ax.set_xlabel("Target Domain", fontsize=20, fontweight="bold")
        ax.set_ylabel("Source Domain", fontsize=20, fontweight="bold")
        ax.tick_params(labelsize=17)

    plt.suptitle("Domain Shift — AUC", fontsize=26, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    save_path = os.path.join(OUTPUT_FOLDER, "heatmap_domain.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Combined PN-top / avg-middle / PU-bottom heatmaps for model shift
# ---------------------------------------------------------------------------

def plot_model_heatmap(df, metrics=None):
    if metrics is None:
        metrics = MODEL_PLOT_METRICS

    sub = df[df["shift_col"] == "model"]
    pn = sub[sub["train_method"] == "PN"]
    pu = sub[(sub["train_method"] == "TEDn") & (sub["source_val"] == "none")]

    all_source_pn = sorted(pn["source_val"].unique())
    all_target = sorted(sub["target_val"].unique())
    n_cols = len(all_target)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for metric in metrics:
        # PN block: source_val × target_val
        pn_point = (
            pn.pivot_table(index="source_val", columns="target_val",
                           values=metric, aggfunc="first")
            .reindex(index=all_source_pn, columns=all_target)
        )

        # Avg. Supervised OOD: mean of off-diagonal PN values per target column
        pn_ood = pn_point.copy().astype(float)
        np.fill_diagonal(pn_ood.values, np.nan)
        avg_row = pn_ood.mean(axis=0, skipna=True).to_frame().T
        avg_row.index = ["Avg. Supervised OOD"]

        # PU + TTA row
        pu_row = pd.DataFrame(np.nan, index=["PU + TTA"], columns=all_target)
        for _, row in pu.iterrows():
            t = row["target_val"]
            if t in all_target:
                pu_row.loc["PU + TTA", t] = row[metric]

        combined = pd.concat([pn_point, avg_row, pu_row])

        # Apply shift for diverging metrics (after reversal already applied in main)
        plot_df = combined.copy().astype(float)
        if metric in DIVERGING_METRICS:
            plot_df = plot_df - 0.5

        # Colormap
        all_vals = [v for v in plot_df.values.flatten() if not pd.isna(v) and not math.isnan(float(v))]
        if metric in BINARY_METRICS:
            cmap = "YlOrBr_r" if metric in REVERSED_METRICS else "YlOrBr"
            data_min, data_max = min(all_vals) if all_vals else 0.0, max(all_vals) if all_vals else 1.0
            margin = max((data_max - data_min) * 0.05, 0.01)
            vmin = max(0.0, data_min - margin)
            vmax = min(1.0, data_max + margin)
            center = (vmin + vmax) / 2
        else:
            cmap = orange_white_purple
            if metric in ("bbe", "plugin-int"):
                center = 0.0
                max_dev = max(abs(v - center) for v in all_vals) if all_vals else 0.5
                vmin, vmax = center - max_dev, center + max_dev
            else:
                data_min, data_max = min(all_vals) if all_vals else 0.0, max(all_vals) if all_vals else 1.0
                margin = max((data_max - data_min) * 0.05, 0.01)
                vmin = max(0.0, data_min - margin)
                vmax = min(1.0, data_max + margin)
                center = 0.5

        annot = plot_df.copy().astype(object)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                val = plot_df.iloc[i, j]
                annot.iloc[i, j] = "" if pd.isna(val) else fmt(val)

        n_pn = len(pn_point)
        n_total = len(plot_df)

        fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_cols), max(6, 2.2 * n_total)))

        sns.heatmap(plot_df, annot=annot, fmt="", cmap=cmap,
                    center=center, vmin=vmin, vmax=vmax, ax=ax,
                    annot_kws={"size": 20})
        ax.collections[0].colorbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: fmt(x))
        )

        # Thick black line between last PN row and Avg. Supervised OOD
        ax.axhline(y=n_pn, color="black", linewidth=4, zorder=4)

        # White space between Avg. Supervised OOD and PU + TTA
        _sep_y = n_total - 1
        _gap_h = 0.20
        ax.add_patch(plt.Rectangle(
            (0, _sep_y - _gap_h / 2), n_cols, _gap_h,
            facecolor="white", edgecolor="none",
            transform=ax.transData, clip_on=True, zorder=3,
        ))
        ax.axhline(y=_sep_y - _gap_h / 2, color="black", linewidth=1, zorder=4)
        ax.axhline(y=_sep_y + _gap_h / 2, color="black", linewidth=1, zorder=4)

        ax.set_xlabel("Target Model", fontsize=20, fontweight="bold")
        ax.set_ylabel("Source Model", fontsize=20, fontweight="bold")
        ax.set_title(f"Model Shift — {NAME_TO_NAME.get(metric, metric)}", fontsize=24, fontweight="bold")
        ax.tick_params(labelsize=17)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_FOLDER, f"heatmap_model_{metric}.pdf")
        plt.savefig(save_path, bbox_inches="tight")
        plt.clf()
        plt.close(fig)
        print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_CSV)
    df = add_accuracy_cols(df)
    df = reverse_bias(reverse_plugin(df))
    df = _deduplicate(df)
    plot_bar_shifts(df)
    plot_domain_heatmaps(df)
    plot_model_heatmap(df)


if __name__ == "__main__":
    main()
