import ast
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

DIR = "."
pattern = f"{DIR}/results_*_oss_val_15_pretrained.csv"
files = sorted(glob.glob(pattern), key=lambda f: int(re.search(r'results_(\d+)_', f).group(1)))

def nan_if_invalid(v):
    return float("nan") if (v is None or v > 1) else v

def mean_ai_score(score_str):
    scores = ast.literal_eval(score_str)
    return sum(scores) / len(scores) if len(scores) else float("nan")

n_samples = None
data = {}  # t -> DataFrame

for fpath in files:
    t = int(re.search(r'results_(\d+)_', fpath).group(1))
    df = pd.read_csv(fpath)
    if n_samples is None:
        n_samples = len(df)
    data[t] = df

ts = sorted(data.keys())

print(ts, n_samples)

# For each sample (mirror), collect values over t
score_by_sample = {i: [] for i in range(n_samples)}
window_ai_by_sample = {i: [] for i in range(n_samples)}
ai_combo_by_sample = {i: [] for i in range(n_samples)}

for t in ts:
    df = data[t]
    score_col = f"mirror_{t}_score_avg"
    for i in range(n_samples):
        score_by_sample[i].append(nan_if_invalid(df.iloc[i][score_col]))
        # window_ai_by_sample[i].append(nan_if_invalid(mean_ai_score(df.iloc[i]["window_ai_assistance_scores"])))
        # ai_combo_by_sample[i].append(nan_if_invalid(df.iloc[i]["fraction_ai"] + 0.5 * df.iloc[i]["fraction_ai_assisted"]))

fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(n_samples):
    ax = axes[i]
    ax.plot(ts, score_by_sample[i], marker='o', label="mirror score avg", color="steelblue")
    # ax.plot(ts, window_ai_by_sample[i], marker='s', label="window_ai_assistance_scores (mean)", color="tomato", linestyle="--")
    # ax.plot(ts, ai_combo_by_sample[i], marker='^', label="frac_ai + 0.5·frac_ai_assisted", color="seagreen", linestyle=":")
    ax.set_title(f"Mirror {i+1}", fontsize=9)
    ax.set_xticks(ts)
    ax.set_ylim(0, 1.05)
    if i % 5 == 0:
        ax.set_ylabel("Score")
    if i >= 10:
        ax.set_xlabel("t")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Mirror score avg vs. window AI assistance score over iterations", y=1.05, fontsize=13)
plt.tight_layout()
plt.savefig(f"{DIR}/plot_preds.png", bbox_inches="tight", dpi=150)
print("Saved plot_preds.png")
plt.show()

# Averaged plot across all 15 mirrors
import numpy as np

avg_score    = [np.nanmean([score_by_sample[i][j]     for i in range(n_samples)]) for j in range(len(ts))]
# avg_window   = [np.nanmean([window_ai_by_sample[i][j] for i in range(n_samples)]) for j in range(len(ts))]
# avg_ai_combo = [np.nanmean([ai_combo_by_sample[i][j]  for i in range(n_samples)]) for j in range(len(ts))]

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(ts, avg_score,    marker='o', label="mirror score avg",                  color="steelblue")
# ax2.plot(ts, avg_window,   marker='s', label="window_ai_assistance_scores (mean)", color="tomato",   linestyle="--")
# ax2.plot(ts, avg_ai_combo, marker='^', label="frac_ai + 0.5·frac_ai_assisted",    color="seagreen",  linestyle=":")
ax2.set_xlabel("t")
ax2.set_ylabel("Score")
ax2.set_xticks(ts)
ax2.set_ylim(0, 1.05)
ax2.legend()
ax2.set_title("Average over all 15 mirrors")
plt.tight_layout()
plt.savefig(f"{DIR}/plot_preds_avg.png", bbox_inches="tight", dpi=150)
print("Saved plot_preds_avg.png")
plt.show()
