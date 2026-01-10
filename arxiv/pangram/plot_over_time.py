import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the results
df = pd.read_parquet('results_0_50.parquet')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Numeric columns to plot
numeric_cols = [
    'fraction_ai', 
    'fraction_ai_assisted', 
    'fraction_human', 
    'num_ai_segments'
]

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    
    # Calculate mean and std for each year-source combination
    grouped = df.groupby(['year', 'source'])[col].agg(['mean', 'std', 'count']).reset_index()
    
    # Plot lines for each source
    for source in df['source'].unique():
        source_data = grouped[grouped['source'] == source]
        
        # Plot mean line
        ax.plot(source_data['year'], source_data['mean'], 
                marker='o', label=source, linewidth=2, markersize=8)
        
        # Add error bars (standard error)
        stderr = source_data['std'] / np.sqrt(source_data['count'])
        ax.fill_between(source_data['year'], 
                        source_data['mean'] - stderr,
                        source_data['mean'] + stderr,
                        alpha=0.2)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{col.replace("_", " ").title()} Over Time by Source', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to only show the years we have
    ax.set_xticks(sorted(df['year'].unique()))

plt.tight_layout()
plt.savefig('ai_detection_metrics_over_time.pdf', dpi=300, bbox_inches='tight')
plt.clf()

# Print summary statistics
print("\n=== Summary Statistics by Year and Source ===\n")
for col in numeric_cols:
    print(f"\n{col.upper()}:")
    summary = df.groupby(['year', 'source'])[col].agg(['mean', 'std', 'count'])
    print(summary.round(4))
    print("-" * 80)

# Check for missing data
print("\n=== Missing Data Report ===")
print(df[numeric_cols].isna().sum())
print(f"\nTotal rows: {len(df)}")
print(f"Rows with any missing numeric data: {df[numeric_cols].isna().any(axis=1).sum()}")