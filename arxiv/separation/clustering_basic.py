"""
Run Linear Discriminant Analysis on Gemini sentence embeddings.

Two tasks:
  Binary     — Human (0) vs. all AI (1)
  Multi-label — Human (0), GPT OSS 120b (1), Llama (2), Gemini 3 (3), Qwen (4)

Outputs per task:
  - Scatter / histogram plot saved as PNG
  - Accuracy + classification report printed to stdout
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

font = {
    'weight': 'bold',
    'size': 20,
}
import matplotlib
matplotlib.rc('font', **font)

EMB_PATH = "/home/kkr36/llm_detection/arxiv/separation/embeddings.npz"
OUT_DIR  = "/home/kkr36/llm_detection/arxiv/separation"

LABEL_NAMES = {
    0: "Human",
    1: "GPT",
    2: "Llama",
    3: "Gemini 3",
    4: "Qwen",
}

COLORS = {
    0: "#2196F3",  # blue
    1: "#F44336",  # red
    2: "#4CAF50",  # green
    3: "#FF9800",  # orange
    4: "#9C27B0",  # purple
}


def plot_centroid_distances(embeddings, labels, path):
    n_labels = len(LABEL_NAMES)
    label_ids = sorted(LABEL_NAMES.keys())
    names = [LABEL_NAMES[i] for i in label_ids]

    centroids = np.array([
        embeddings[labels == i].mean(axis=0) for i in label_ids
    ])

    dist_matrix = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    human_idx = names.index("Human")

    avg_offdiag = np.array([
        np.nanmean([
            dist_matrix[i, j]
            for i in range(n_labels)
            if i != j and i != human_idx
        ])
        for j in range(n_labels)
    ])

    fig = plt.figure(figsize=(8, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[n_labels, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax_avg = fig.add_subplot(gs[1])

    im = ax.imshow(dist_matrix, cmap="viridis")

    ax.set_xticks([])
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(names, fontsize=13)

    ax.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    for i in range(n_labels):
        for j in range(n_labels):
            ax.text(j, i, f"{dist_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if dist_matrix[i, j] < dist_matrix.max() * 0.6 else "black")

    ax_avg.imshow(avg_offdiag[np.newaxis, :], cmap="viridis",
                  vmin=dist_matrix.min(), vmax=dist_matrix.max())
    ax_avg.set_xticks(range(n_labels))
    ax_avg.set_xticklabels(names, rotation=45, ha="right", fontsize=13)
    ax_avg.set_yticks([0])
    ax_avg.set_yticklabels(["Avg\nOff-Diag\nLLM"], fontsize=11)
    ax_avg.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax_avg.set_yticks(np.arange(-0.5, 1, 1), minor=True)
    ax_avg.grid(which="minor", color="white", linewidth=1.5)
    ax_avg.tick_params(which="minor", length=0)
    for j in range(n_labels):
        ax_avg.text(j, 0, f"{avg_offdiag[j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if avg_offdiag[j] < dist_matrix.max() * 0.6 else "black")

    cbar = fig.colorbar(im, ax=[ax, ax_avg])
    cbar.set_label("Euclidean Distance", fontsize=13)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_linear_boundary_errors(embeddings, labels, path):
    n_labels = len(LABEL_NAMES)
    label_ids = sorted(LABEL_NAMES.keys())
    names = [LABEL_NAMES[i] for i in label_ids]

    error_matrix = np.full((n_labels, n_labels), np.nan)

    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                continue
            mask = (labels == label_ids[i]) | (labels == label_ids[j])
            X = embeddings[mask]
            y = (labels[mask] == label_ids[j]).astype(int)

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X, y)
            error_matrix[i, j] = 1.0 - clf.score(X, y)

    vmin, vmax = np.nanmin(error_matrix), np.nanmax(error_matrix)

    human_idx = names.index("Human")

    avg_offdiag = np.array([
        np.nanmean([
            error_matrix[i, j]
            for i in range(n_labels)
            if i != j and i != human_idx
        ])
        for j in range(n_labels)
    ])

    fig = plt.figure(figsize=(8, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[n_labels, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax_avg = fig.add_subplot(gs[1])

    masked = np.ma.masked_invalid(error_matrix)
    im = ax.imshow(masked, cmap="Reds", vmin=vmin, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(names, fontsize=13)

    ax.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=14)
            else:
                val = error_matrix[i, j]
                ax.text(j, i, f"{val:.3f}",
                        ha="center", va="center", fontsize=11,
                        color="white" if val > (vmin + (vmax - vmin) * 0.6) else "black")

    ax_avg.imshow(avg_offdiag[np.newaxis, :], cmap="Reds", vmin=vmin, vmax=vmax)
    ax_avg.set_xticks(range(n_labels))
    ax_avg.set_xticklabels(names, rotation=45, ha="right", fontsize=13)
    ax_avg.set_yticks([0])
    ax_avg.set_yticklabels(["Avg\nOff-Diag\nLLM"], fontsize=11)
    ax_avg.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax_avg.set_yticks(np.arange(-0.5, 1, 1), minor=True)
    ax_avg.grid(which="minor", color="white", linewidth=1.5)
    ax_avg.tick_params(which="minor", length=0)
    for j in range(n_labels):
        ax_avg.text(j, 0, f"{avg_offdiag[j]:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if avg_offdiag[j] > (vmin + (vmax - vmin) * 0.6) else "black")

    cbar = fig.colorbar(im, ax=[ax, ax_avg])
    cbar.set_label("Error Rate", fontsize=13)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")



def main():
    data = np.load(EMB_PATH)
    embeddings = data["embeddings"]
    labels     = data["labels"]

    print(f"Loaded: {embeddings.shape[0]} sentences, {embeddings.shape[1]}-dim embeddings")
    for lbl, name in LABEL_NAMES.items():
        print(f"  label {lbl} ({name}): {(labels == lbl).sum()}")

    plot_centroid_distances(
        embeddings, labels,
        os.path.join(OUT_DIR, "centroid_distances.pdf")
    )

    plot_linear_boundary_errors(
        embeddings, labels,
        os.path.join(OUT_DIR, "linear_boundary_errors.pdf")
    )


if __name__ == "__main__":
    main()
