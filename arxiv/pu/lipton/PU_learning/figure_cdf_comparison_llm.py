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


def plot_ecdf(ax, x, label, color):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", label=label, color=color)


fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.3))

# Left: PN, train=2010, test=[2010, 2016, 2020]
for year in YEARS:
    probs, targets = load_preds(PRED_ROOT / "PN" / "train_2010" / f"test_{year}")
    mask = targets == 0
    plot_ecdf(axes[0], probs[mask], label=str(year), color=YEAR_COLORS[year])

# Right: TEDn, train=test=year
for year in YEARS:
    probs, targets = load_preds(PRED_ROOT / "TEDn" / f"train_{year}" / f"test_{year}")
    mask = targets == 0
    plot_ecdf(axes[1], probs[mask], label=str(year), color=YEAR_COLORS[year])

for ax in axes:
    ax.set_xlabel("P(LLM | LLM)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: "" if v == 0 else f"{v:g}")
    )

axes[0].text(0.5, 0.95, "Supervised 2010", transform=axes[0].transAxes,
             ha="center", va="top", fontsize=17, fontweight="bold")
axes[1].text(0.5, 0.95, "PU with\nTest-Time Adaptation", transform=axes[1].transAxes,
             ha="center", va="top", fontsize=17, fontweight="bold", linespacing=1.3)

# Shared legend above the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.05), frameon=False)

plt.tight_layout()
plt.savefig("figure_cdf_comparison_llm.pdf", format="pdf", bbox_inches="tight")
print("Saved: figure_cdf_comparison_llm.pdf")
