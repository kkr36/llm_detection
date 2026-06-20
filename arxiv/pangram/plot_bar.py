import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import ast

# Read the results
# df = pd.read_csv('results_everything_100_2010.csv')
df = pd.read_csv('results_everything_100_2010_5142026.csv')

# Parse window_ai_assistance_score (stored as list or stringified list) → row average
def parse_score_list(val):
    if isinstance(val, list):
        scores = val
    else:
        try:
            scores = ast.literal_eval(str(val))
        except (ValueError, SyntaxError):
            return np.nan
    if not scores:
        return np.nan
    return float(np.mean([s for s in scores if s is not None]))

df['avg_window_ai_assistance_score'] = df['window_ai_assistance_scores'].apply(parse_score_list)

df['source'] = df['source'].replace('ChatGPT 5.4', 'GPT 5.4 March 2026')
df['source'] = df['source'].replace('ChatGPT 5.4 new', 'GPT 5.4 May 2026')
df['source'] = df['source'].replace('GPT OSS 120b', 'gpt-oss-120b')
df['source'] = df['source'].replace('GPT OSS 20b', 'gpt-oss-20b')
df['source'] = df['source'].replace('Llama 3.3 70b Instruct', 'Llama 3.3 70B Instruct')
df['source'] = df['source'].replace('Gemini 3 Preview', 'Gemini 3 Pro Preview')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Metrics to analyze
numeric_cols = [
    'fraction_ai',
    'fraction_ai_assisted',
    'fraction_human',
    'num_ai_segments',
    'avg_window_ai_assistance_score',
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
# Plot helpers
# =========================
def _plot_grouped_bars(ax, cols, sources, means, cis, palette):
    n_src = len(sources)
    x = np.arange(len(cols))
    bar_width = 0.8 / n_src
    for i, source in enumerate(sources):
        offsets = x + (i - (n_src - 1) / 2) * bar_width
        heights = means.loc[source, cols].values
        errors  = cis.loc[source, cols].values
        ax.bar(offsets, heights, width=bar_width, color=palette[i],
               label=source.replace("human_abstract", "Human"), alpha=0.85, zorder=3)
        ax.errorbar(offsets, heights, yerr=errors, fmt='none', ecolor='black',
                    elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
    tick_labels = [c.replace("_", " ").replace("avg ", "Avg ").title().replace("Ai", "AI") for c in cols]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_ylabel("Mean Value", fontsize=17)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis='y', which='major', linewidth=0.8, zorder=0)
    ax.grid(axis='y', which='minor', linewidth=0.4, linestyle=':', zorder=0)
    ax.set_axisbelow(True)
    n_cols_legend = min(n_src, (n_src + 1) // 2)
    ax.legend(title="Source", title_fontsize=13, fontsize=12,
              loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=n_cols_legend, framealpha=0.9)


def plot_fraction_bars(sources, means, cis, palette):
    cols = ['fraction_ai', 'fraction_ai_assisted', 'fraction_human']
    fig, ax = plt.subplots(figsize=(10, 7))
    _plot_grouped_bars(ax, cols, sources, means, cis, palette)
    # ax.set_title("AI/Assisted/Human Fractions by LLM Source\n(Mean ± 95% CI)",
    #              fontsize=16, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig('ai_detection_fractions.pdf', dpi=300, bbox_inches='tight')
    plt.clf()
    print("Saved: ai_detection_fractions.pdf")


def plot_segment_score_bars(sources, means, cis, palette):
    cols = ['num_ai_segments', 'avg_window_ai_assistance_score']
    fig, ax = plt.subplots(figsize=(10, 7))
    _plot_grouped_bars(ax, cols, sources, means, cis, palette)
    # ax.set_title("AI Segments & Avg Window Score by LLM Source\n(Mean ± 95% CI)",
                #  fontsize=16, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig('ai_detection_segments_score.pdf', dpi=300, bbox_inches='tight')
    plt.clf()
    print("Saved: ai_detection_segments_score.pdf")


# =========================
# Plot (original — all metrics combined)
# =========================
fig, ax = plt.subplots()

for i, source in enumerate(sources):
    # import pdb; pdb.set_trace()
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
tick_labels = [m.replace("_", " ").replace("avg ", "Avg ").title() for m in numeric_cols]
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=11)
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

plot_fraction_bars(sources, means, cis, palette)
plot_segment_score_bars(sources, means, cis, palette)

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