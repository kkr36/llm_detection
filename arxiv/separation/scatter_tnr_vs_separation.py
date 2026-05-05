"""
Scatter plots: PU-learning TNR vs. embedding-space separation metrics.

Each dot = ordered pair (train_llm=a, test_llm=b).
  - x-axis (plot 1): centroid distance between a and b
  - x-axis (plot 2): logistic-regression error rate between a and b
  - y-axis (both):   TNR from logging_accuracy_llm.csv (PN method)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression

matplotlib.rc('font', **{'weight': 'bold', 'size': 14})

EMB_PATH = "/home/kkr36/llm_detection/arxiv/separation/embeddings.npz"
CSV_PATH = "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/logging_accuracy_llm.csv"
OUT_DIR  = "/home/kkr36/llm_detection/arxiv/separation"

# Only AI LLMs (skip Human label 0)
LABEL_NAMES = {
    1: "GPT OSS 120b",
    2: "Llama 3.3 70b",
    3: "Gemini 3 Preview",
    4: "Qwen",
}

# Map from CSV name → embedding label id
CSV_TO_LABEL = {
    "GPT OSS 120b":           1,
    "Llama 3.3 70b Instruct": 2,
    "Gemini 3 Preview":       3,
    "Qwen":                   4,
}

COLORS = {
    "GPT OSS 120b":     "#F44336",
    "Llama 3.3 70b":    "#4CAF50",
    "Gemini 3 Preview": "#FF9800",
    "Qwen":             "#9C27B0",
}

MARKER_SAME = "D"   # diamond for same-LLM pairs
MARKER_DIFF = "o"   # circle for cross-LLM pairs


def compute_centroid_dist(embeddings, labels):
    label_ids = sorted(LABEL_NAMES.keys())
    centroids = {i: embeddings[labels == i].mean(axis=0) for i in label_ids}
    dist = {}
    for i in label_ids:
        for j in label_ids:
            dist[(i, j)] = np.linalg.norm(centroids[i] - centroids[j])
    return dist


def compute_error_rate(embeddings, labels):
    label_ids = sorted(LABEL_NAMES.keys())
    err = {}
    for i in label_ids:
        for j in label_ids:
            if i == j:
                err[(i, j)] = 0.0
                continue
            mask = (labels == i) | (labels == j)
            X = embeddings[mask]
            y = (labels[mask] == j).astype(int)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X, y)
            err[(i, j)] = 1.0 - clf.score(X, y)
    return err


def build_averaged_pairs(df, centroid_dist, error_rate):
    """
    For each unordered pair {a, b}, average the TNR values of (a→b) and (b→a)
    from the CSV, and average the error rate (centroid distance is symmetric).
    Same-LLM pairs (a == b) have only one row so no averaging is needed.
    Returns (pairs, x_centroid, x_error, y_tnr) where each entry is one
    unordered pair represented as (name_a, name_b) with name_a <= name_b.
    """
    # Accumulate per unordered pair
    from collections import defaultdict
    buckets = defaultdict(lambda: {"tnr": [], "centroid": [], "error": []})

    for _, row in df.iterrows():
        train_csv = row['train_llm']
        test_csv  = row['test_llm']
        if train_csv not in CSV_TO_LABEL or test_csv not in CSV_TO_LABEL:
            continue
        if train_csv == test_csv:
            continue
        i = CSV_TO_LABEL[train_csv]
        j = CSV_TO_LABEL[test_csv]
        name_a = LABEL_NAMES[i]
        name_b = LABEL_NAMES[j]
        key = tuple(sorted([name_a, name_b]))

        buckets[key]["tnr"].append(row['tnr'])
        buckets[key]["centroid"].append(centroid_dist[(i, j)])
        buckets[key]["error"].append(error_rate[(i, j)])

    pairs, x_centroid, x_error, y_tnr = [], [], [], []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        pairs.append(key)
        x_centroid.append(np.mean(b["centroid"]))
        x_error.append(np.mean(b["error"]))
        y_tnr.append(np.mean(b["tnr"]))

    return pairs, x_centroid, x_error, y_tnr


# Palette for unordered pairs (one color per pair, assigned in sorted order)
_PAIR_COLORS = [
    "#E53935", "#8E24AA", "#1E88E5", "#00897B",
    "#F4511E", "#FFB300", "#43A047", "#6D4C41",
    "#039BE5", "#757575",
]


def make_scatter_averaged(pairs, x_vals, y_vals, x_label, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    for idx, ((name_a, name_b), x, y) in enumerate(zip(pairs, x_vals, y_vals)):
        color  = _PAIR_COLORS[idx % len(_PAIR_COLORS)]
        marker = MARKER_SAME if name_a == name_b else MARKER_DIFF
        ax.scatter(x, y, color=color, marker=marker, s=120, zorder=3,
                   edgecolors="black", linewidths=0.6)
        label = f"({name_a[:3]},{name_b[:3]})" if name_a != name_b else f"({name_a[:3]})"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 3),
                    fontsize=12, fontweight='normal')

    m, b = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(min(x_vals), max(x_vals), 200)
    ax.plot(x_line, m * x_line + b, color='red', linestyle=':', linewidth=2.5, zorder=4)

    legend_handles = [
        matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=_PAIR_COLORS[idx % len(_PAIR_COLORS)],
                                 markersize=9, markeredgecolor='black',
                                 label=(f"{a[:3]}–{b[:3]}" if a != b else a[:3]))
        for idx, (a, b) in enumerate(pairs)
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc='best',
              title="pair", title_fontsize=12)

    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel("Avg AI Recall", fontweight='bold')
    # ax.set_title(f"Avg TNR vs. {x_label} (averaged pairs)", fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def make_scatter(pairs, x_vals, y_vals, x_label, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    for (train_name, test_name), x, y in zip(pairs, x_vals, y_vals):
        color = COLORS[test_name]
        marker = MARKER_SAME if train_name == test_name else MARKER_DIFF
        ax.scatter(x, y, color=color, marker=marker, s=100, zorder=3,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(
            f"({train_name[:3]},{test_name[:3]})",
            (x, y), textcoords="offset points", xytext=(5, 3),
            fontsize=12, fontweight='normal'
        )

    m, b = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(min(x_vals), max(x_vals), 200)
    ax.plot(x_line, m * x_line + b, color='red', linestyle=':', linewidth=2.5, zorder=4)

    # Legend for colors (test LLM)
    legend_handles = [
        matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=c, markersize=9,
                                 markeredgecolor='black', label=name)
        for name, c in COLORS.items()
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc='best',
              title="test LLM", title_fontsize=12)

    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel("AI Recall", fontweight='bold')
    # ax.set_title(f"TNR vs. {x_label}", fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    # Load embeddings
    data = np.load(EMB_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]

    # Compute separation metrics
    print("Computing centroid distances...")
    centroid_dist = compute_centroid_dist(embeddings, labels)

    print("Computing logistic-regression error rates...")
    error_rate = compute_error_rate(embeddings, labels)

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df = df[(df['train_llm'] != 'all') & (df['test_llm'] != 'all') &
            (df['learning_method'] == 'PN')]

    # Build aligned lists
    pairs, x_centroid, x_error, y_tnr = [], [], [], []

    for _, row in df.iterrows():
        train_csv = row['train_llm']
        test_csv  = row['test_llm']
        if train_csv not in CSV_TO_LABEL or test_csv not in CSV_TO_LABEL:
            continue
        if train_csv == test_csv:
            continue
        i = CSV_TO_LABEL[train_csv]
        j = CSV_TO_LABEL[test_csv]
        train_name = LABEL_NAMES[i]
        test_name  = LABEL_NAMES[j]

        pairs.append((train_name, test_name))
        x_centroid.append(centroid_dist[(i, j)])
        x_error.append(error_rate[(i, j)])
        y_tnr.append(row['tnr'])

    print(f"\nPlotting {len(pairs)} ordered pairs:")
    for (a, b), xc, xe, y in zip(pairs, x_centroid, x_error, y_tnr):
        print(f"  ({a}, {b}): centroid_dist={xc:.3f}, error={xe:.4f}, tnr={y:.4f}")

    make_scatter(pairs, x_centroid, y_tnr,
                 "Centroid Distance",
                 os.path.join(OUT_DIR, "scatter_tnr_vs_centroid_dist.pdf"))

    make_scatter(pairs, x_error, y_tnr,
                 "Logistic Regression Error Rate",
                 os.path.join(OUT_DIR, "scatter_tnr_vs_error_rate.pdf"))

    # Averaged-pair variants: collapse (a,b)/(b,a) by averaging TNR and error rate
    avg_pairs, avg_x_centroid, avg_x_error, avg_y_tnr = build_averaged_pairs(
        df, centroid_dist, error_rate
    )

    print(f"\nPlotting {len(avg_pairs)} averaged unordered pairs:")
    for (a, b), xc, xe, y in zip(avg_pairs, avg_x_centroid, avg_x_error, avg_y_tnr):
        print(f"  {{{a}, {b}}}: centroid_dist={xc:.3f}, error={xe:.4f}, avg_tnr={y:.4f}")

    make_scatter_averaged(avg_pairs, avg_x_centroid, avg_y_tnr,
                          "Centroid Distance",
                          os.path.join(OUT_DIR, "scatter_avgtnr_vs_centroid_dist.pdf"))

    make_scatter_averaged(avg_pairs, avg_x_error, avg_y_tnr,
                          "Logistic Regression Error Rate",
                          os.path.join(OUT_DIR, "scatter_avgtnr_vs_error_rate.pdf"))


if __name__ == "__main__":
    main()
