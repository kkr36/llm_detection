import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

font = {
    'weight': 'bold',
    'size': 20,
}
import matplotlib
matplotlib.rc('font', **font)

EMB_PATH = "/home/kkr36/llm_detection/arxiv/separation/embeddings_gemini_full.npz"
OUT_DIR  = "/home/kkr36/llm_detection/arxiv/separation"

LABEL_NAMES = {
    0: "Human",
    1: "Gemini 2.0 Flash-Lite",
    2: "Gemini 3 Preview",
    3: "Gemini 2.0 Flash",
    4: "Gemini 2.5 Flash",
    5: "Gemini 2.5 Pro",
}

COLORS = {
    0: "#2196F3",  # blue
    1: "#F44336",  # red
    2: "#4CAF50",  # green
    3: "#FF9800",  # orange
    4: "#9C27B0",  # purple
    5: "#00BCD4",  # cyan
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

    vmin, vmax = dist_matrix.min(), dist_matrix.max()

    # --- off-diagonal column averages ---
    human_idx = names.index("Human")

    avg_offdiag = np.array([
        np.nanmean([
            dist_matrix[i, j]
            for i in range(n_labels)
            if i != j and i != human_idx
        ])
        for j in range(n_labels)
    ])

    # --- layout with whitespace ---
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[n_labels, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax_avg = fig.add_subplot(gs[1])

    im = ax.imshow(dist_matrix, cmap="viridis", vmin=vmin, vmax=vmax)

    # --- main heatmap ---
    ax.set_xticks([])
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(names, fontsize=12)

    ax.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # --- annotate matrix ---
    for i in range(n_labels):
        for j in range(n_labels):
            val = dist_matrix[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=10,
                color="white" if val < vmax * 0.6 else "black"
            )

    # --- avg row ---
    ax_avg.imshow(avg_offdiag[np.newaxis, :], cmap="viridis", vmin=vmin, vmax=vmax)
    ax_avg.set_xticks(range(n_labels))
    ax_avg.set_xticklabels(names, rotation=45, ha="right", fontsize=12)
    ax_avg.set_yticks([0])
    ax_avg.set_yticklabels(["Avg\nOff-Diag\nLLM"], fontsize=11)

    ax_avg.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax_avg.set_yticks(np.arange(-0.5, 1, 1), minor=True)
    ax_avg.grid(which="minor", color="white", linewidth=1.5)
    ax_avg.tick_params(which="minor", length=0)

    for j in range(n_labels):
        val = avg_offdiag[j]
        ax_avg.text(
            j, 0, f"{val:.2f}",
            ha="center", va="center", fontsize=10,
            color="white" if val < vmax * 0.6 else "black"
        )

    # --- shared colorbar ---
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

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)
            error_matrix[i, j] = 1.0 - clf.score(X_te, y_te)

    vmin, vmax = np.nanmin(error_matrix), np.nanmax(error_matrix)

    # --- off-diagonal column averages ---
    human_idx = names.index("Human")

    avg_offdiag = np.array([
        np.nanmean([
            error_matrix[i, j]
            for i in range(n_labels)
            if i != j and i != human_idx
        ])
        for j in range(n_labels)
    ])

    # --- layout with whitespace ---
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[n_labels, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    ax_avg = fig.add_subplot(gs[1])

    masked = np.ma.masked_invalid(error_matrix)
    im = ax.imshow(masked, cmap="Reds", vmin=vmin, vmax=vmax)

    # --- main heatmap ticks ---
    ax.set_xticks([])
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(names, fontsize=12)

    ax.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # --- annotate matrix ---
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=14)
            else:
                val = error_matrix[i, j]
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if val > (vmin + (vmax - vmin) * 0.6) else "black"
                )

    # --- avg row ---
    ax_avg.imshow(avg_offdiag[np.newaxis, :], cmap="Reds", vmin=vmin, vmax=vmax)
    ax_avg.set_xticks(range(n_labels))
    ax_avg.set_xticklabels(names, rotation=45, ha="right", fontsize=12)
    ax_avg.set_yticks([0])
    ax_avg.set_yticklabels(["Avg\nOff-Diag\nLLM"], fontsize=11)

    ax_avg.set_xticks(np.arange(-0.5, n_labels, 1), minor=True)
    ax_avg.set_yticks(np.arange(-0.5, 1, 1), minor=True)
    ax_avg.grid(which="minor", color="white", linewidth=1.5)
    ax_avg.tick_params(which="minor", length=0)

    for j in range(n_labels):
        val = avg_offdiag[j]
        ax_avg.text(
            j, 0, f"{val:.3f}",
            ha="center", va="center", fontsize=10,
            color="white" if val > (vmin + (vmax - vmin) * 0.6) else "black"
        )

    # --- shared colorbar ---
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
        os.path.join(OUT_DIR, "centroid_distances_gemini.pdf")
    )

    plot_linear_boundary_errors(
        embeddings, labels,
        os.path.join(OUT_DIR, "linear_boundary_errors_gemini.pdf")
    )


if __name__ == "__main__":
    main()
