import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20
    }
import matplotlib
matplotlib.rc('font', **font)
CSV_PATHS = [
    "results_0_oss_test_50_pretrained.csv",
    "results_4_oss_test_50_pretrained.csv",
]

mirror_scores = []
mirror_errs = []
ai_combined_scores = []
ai_combined_errs = []
labels = []

for path in CSV_PATHS:
    match = re.search(r"results_(\d+)_", path)
    i = match.group(1)

    df = pd.read_csv(path)
    n = len(df)

    mirror_col = f"mirror_{i}_score_avg"
    mirror_mean = df[mirror_col].mean()
    mirror_se = df[mirror_col].std() / np.sqrt(n)

    ai_combined = df["fraction_ai"] + 0.5 * df["fraction_ai_assisted"]
    ai_combined_mean = ai_combined.mean()
    ai_combined_se = ai_combined.std() / np.sqrt(n)

    mirror_scores.append(mirror_mean)
    mirror_errs.append(mirror_se)
    ai_combined_scores.append(ai_combined_mean)
    ai_combined_errs.append(ai_combined_se)
    # labels.append(f"i={i}")
    label = "Original\nPrompt" if i == '0' else "Optimized\nPrompt"
    print(i, label)
    labels.append(label)
print(labels)
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width / 2, mirror_scores, width, yerr=mirror_errs, capsize=4, label="Our Model")
ax.bar(x + width / 2, ai_combined_scores, width, yerr=ai_combined_errs, capsize=4, label="Pangram")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean Estimated P(LLM)")
ax.legend()
plt.tight_layout()
plt.savefig("bar_plot.pdf", format="pdf", bbox_inches="tight")
plt.show()
