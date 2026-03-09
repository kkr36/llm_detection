import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# Read the results
df = pd.read_csv('results_everything_100_2010.csv')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Metrics to analyze
numeric_cols = [
    'fraction_ai',
    'fraction_ai_assisted',
    'fraction_human',
    'num_ai_segments'
]

# =========================
# Aggregate by source (mean + 95% CI)
# =========================
def ci95(x):
    n = x.count()
    if n < 2:
        return np.nan
    return 1.96 * x.std() / np.sqrt(n)

summary = df.groupby('source')[numeric_cols].agg(['mean', ci95, 'count'])
means   = summary.xs('mean', axis=1, level=1)
cis     = summary.xs('ci95', axis=1, level=1)

sources = means.index.tolist()
n_sources  = len(sources)
n_metrics  = len(numeric_cols)

x = np.arange(n_metrics)
bar_width = 0.8 / n_sources          # keep total bar group width ≤ 0.8

# Colour palette — one colour per source
palette = sns.color_palette("tab10", n_sources)

# =========================
# Plot
# =========================
fig, ax = plt.subplots()

for i, source in enumerate(sources):
    offsets = x + (i - (n_sources - 1) / 2) * bar_width
    heights = means.loc[source].values
    errors  = cis.loc[source].values

    ax.bar(
        offsets,
        heights,
        width=bar_width,
        color=palette[i],
        label=source,
        alpha=0.85,
        zorder=3,
    )

    ax.errorbar(
        offsets,
        heights,
        yerr=errors,
        fmt='none',
        ecolor='black',
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=4,
    )

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", " ").title() for m in numeric_cols], fontsize=12)
ax.set_ylabel("Mean Value", fontsize=12)
ax.set_title(
    "AI Detection Metrics by LLM Source\n(Mean ± 95% CI)",
    fontsize=15,
    fontweight="bold",
    pad=14,
)

ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis='y', which='major', linewidth=0.8, zorder=0)
ax.grid(axis='y', which='minor', linewidth=0.4, linestyle=':', zorder=0)
ax.set_axisbelow(True)

ax.legend(
    title="Source",
    title_fontsize=11,
    fontsize=10,
    loc='upper right',
    framealpha=0.9,
)

plt.tight_layout()
plt.savefig('ai_detection_grouped_bars.pdf', dpi=300, bbox_inches='tight')
plt.clf()
print("Saved: ai_detection_grouped_bars.pdf")

# =========================
# Print summary statistics
# =========================
print("\n=== Summary Statistics by Source ===\n")

for col in numeric_cols:
    print(f"\n{col.upper()}:")
    col_summary = df.groupby('source')[col].agg(['mean', 'std', 'count'])
    col_summary['95% CI'] = 1.96 * col_summary['std'] / np.sqrt(col_summary['count'])
    print(col_summary.round(4))
    print("-" * 80)

# =========================
# Missing data report
# =========================
print("\n=== Missing Data Report ===")
print(df[numeric_cols].isna().sum())
print(f"\nTotal rows: {len(df)}")
print(f"Rows with any missing numeric data: {df[numeric_cols].isna().any(axis=1).sum()}")