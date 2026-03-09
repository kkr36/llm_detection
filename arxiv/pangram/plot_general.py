import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the results
df = pd.read_parquet('results_54_0_100.parquet')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 12)

# Metrics to analyze
numeric_cols = [
    'fraction_ai',
    'fraction_ai_assisted',
    'fraction_human',
    'num_ai_segments'
]

# =========================
# Aggregate by source
# =========================
summary = df.groupby('source')[numeric_cols].agg(['mean', 'std', 'count'])

means = summary.xs('mean', axis=1, level=1)

# =========================
# Normalize metrics for radar chart
# =========================
norm_means = (means - means.min()) / (means.max() - means.min())

# =========================
# Radar chart setup
# =========================
metrics = numeric_cols
num_metrics = len(metrics)

angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(subplot_kw=dict(polar=True))

for source in norm_means.index:
    values = norm_means.loc[source].tolist()
    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=2,
        label=source
    )

    ax.fill(
        angles,
        values,
        alpha=0.1
    )

# Axis formatting
ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])

ax.set_title(
    "AI Detection Metric Profile by LLM Source",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

plt.tight_layout()
plt.savefig('ai_detection_radar_by_source.pdf', dpi=300, bbox_inches='tight')
plt.clf()

# =========================
# Print summary statistics
# =========================
print("\n=== Summary Statistics by Source ===\n")

for col in numeric_cols:
    print(f"\n{col.upper()}:")
    col_summary = df.groupby('source')[col].agg(['mean', 'std', 'count'])
    print(col_summary.round(4))
    print("-" * 80)

# =========================
# Missing data report
# =========================
print("\n=== Missing Data Report ===")

print(df[numeric_cols].isna().sum())

print(f"\nTotal rows: {len(df)}")
print(f"Rows with any missing numeric data: {df[numeric_cols].isna().any(axis=1).sum()}")