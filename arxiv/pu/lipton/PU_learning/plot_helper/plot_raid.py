import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', weight='bold', size=20)

INPUT_CSV = "../logging_accuracy_raid_cross.csv"
OUTPUT_PDF = "raid_grouped_bar.pdf"

METRIC = "auc"
CI = "0.95"

METHOD_DISPLAY = {
    "PN":   "Supervised",
    "TEDn": "TTA + PU",
    "PNU":  "TTA + PNU",
}
METHOD_ORDER = ["PN", "TEDn", "PNU"]
COLORS = {
    "PN":   "red",
    "TEDn": "steelblue",
    "PNU":  "darkorange",
}

ATTACK_DISPLAY = {
    "all":              "All",
    "article_deletion": "Article\nDeletion",
    "homoglyph":        "Homoglyph",
    "paraphrase":       "Paraphrase",
    "whitespace":       "Whitespace",
}
ATTACK_ORDER = ["all", "article_deletion", "homoglyph", "paraphrase", "whitespace"]


def main():
    df = pd.read_csv(INPUT_CSV)

    # keep only attacks that exist for all 3 methods
    df = df[df["test_attack"].isin(ATTACK_ORDER)]

    n_attacks = len(ATTACK_ORDER)
    n_methods = len(METHOD_ORDER)
    bar_width = 0.22
    group_gap = 0.1
    group_width = n_methods * bar_width + group_gap
    x_centers = np.arange(n_attacks) * group_width

    fig, ax = plt.subplots(figsize=(12, 5))

    for m_idx, method in enumerate(METHOD_ORDER):
        subset = df[df["train_method"] == method].set_index("test_attack")
        heights, errs_lo, errs_hi = [], [], []
        for attack in ATTACK_ORDER:
            if attack in subset.index:
                row = subset.loc[attack]
                val = row[METRIC]
                lo  = val - row[f"{METRIC}_l_{CI}"]
                hi  = row[f"{METRIC}_u_{CI}"] - val
            else:
                val, lo, hi = 0.0, 0.0, 0.0
            heights.append(val)
            errs_lo.append(lo)
            errs_hi.append(hi)

        offset = (m_idx - (n_methods - 1) / 2) * bar_width
        x_pos = x_centers + offset
        ax.bar(
            x_pos, heights,
            width=bar_width,
            label=METHOD_DISPLAY[method],
            color=COLORS[method],
            alpha=0.85,
            zorder=3,
        )
        ax.errorbar(
            x_pos, heights,
            yerr=[errs_lo, errs_hi],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            zorder=4,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [ATTACK_DISPLAY[a] for a in ATTACK_ORDER],
        fontsize=17, fontweight="bold",
    )
    ax.set_ylabel("AUC", fontsize=20, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower right",
        fontsize=16,
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), OUTPUT_PDF), bbox_inches="tight")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
