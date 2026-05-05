import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

font = {'weight': 'bold', 'size': 20}
matplotlib.rc('font', **font)

PRED_ROOT = Path("/share/garg/arxiv_kaggle/predictions/temporal")
YEARS = [2010, 2016, 2020]
YEAR_COLORS = {2010: "blue", 2016: "purple", 2020: "red"}
SEEDS = list(range(10))


def load_preds(pred_dir):
    probs, targets = [], []
    for seed in SEEDS:
        d = np.load(pred_dir / f"seed_{seed}.npz")
        probs.append(d["unlabeled_probs"][:, 0])
        targets.append(d["unlabeled_targets"])
    return np.concatenate(probs), np.concatenate(targets)


fig, axes = plt.subplots(2, 2, figsize=(9.3, 8.6))

# Row 0: P(LLM | LLM)  (targets == 0, probs[:, 0])
for year in YEARS:
    probs, targets = load_preds(PRED_ROOT / "PN" / "train_2010" / f"test_{year}")
    mask = targets == 0
    axes[0, 0].hist(probs[mask], bins=50, density=True, alpha=0.5,
                    label=str(year), color=YEAR_COLORS[year])

for year in YEARS:
    probs, targets = load_preds(PRED_ROOT / "TEDn" / f"train_{year}" / f"test_{year}")
    mask = targets == 0
    axes[0, 1].hist(probs[mask], bins=50, density=True, alpha=0.5,
                    label=str(year), color=YEAR_COLORS[year])

# Row 1: P(human | human)  (targets == 1, 1 - probs[:, 0])
for year in YEARS:
    probs, targets = load_preds(PRED_ROOT / "PN" / "train_2010" / f"test_{year}")
    mask = targets == 1
    axes[1, 0].hist(1 - probs[mask], bins=50, density=True, alpha=0.5,
                    label=str(year), color=YEAR_COLORS[year])

for year in YEARS:
    probs, targets = load_preds(PRED_ROOT / "TEDn" / f"train_{year}" / f"test_{year}")
    mask = targets == 1
    axes[1, 1].hist(1 - probs[mask], bins=50, density=True, alpha=0.5,
                    label=str(year), color=YEAR_COLORS[year])

for ax in axes[0]:
    ax.set_xlabel("P(LLM | LLM)")
for ax in axes[1]:
    ax.set_xlabel("P(human | human)")
for ax in axes.flat:
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "" if v == 0 else f"{v:g}")
    )

for ax, title in zip(axes[0], ["Supervised", "PU with\nTest-Time Adaptation"]):
    ax.text(0.5, 0.95, title, transform=ax.transAxes,
            ha="center", va="top", fontsize=17, fontweight="bold", linespacing=1.3)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.06), frameon=False)

plt.tight_layout()
plt.savefig("figure_hist_comparison.pdf", format="pdf", bbox_inches="tight")
print("Saved: figure_hist_comparison.pdf")
