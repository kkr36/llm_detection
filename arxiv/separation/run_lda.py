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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20
    }
import matplotlib
matplotlib.rc('font', **font)
# plt.rcParams.update({"font.size": 14})

EMB_PATH = "/home/kkr36/llm_detection/arxiv/separation/embeddings.npz"
OUT_DIR  = "/home/kkr36/llm_detection/arxiv/separation"

LABEL_NAMES = {
    0: "Human",
    1: "GPT OSS 120b",
    2: "Llama 3.3 70b",
    3: "Gemini 3 Preview",
    4: "Qwen",
}

COLORS = {
    0: "#2196F3",  # blue
    1: "#F44336",  # red
    2: "#4CAF50",  # green
    3: "#FF9800",  # orange
    4: "#9C27B0",  # purple
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_binary(embeddings, y_binary, lda, path):
    """1-D LDA score histogram: Human vs. AI."""
    scores = lda.transform(embeddings).flatten()

    fig, ax = plt.subplots(figsize=(9, 4))
    for lbl, name, color in [(0, "Human", COLORS[0]), (1, "AI (all)", "#F44336")]:
        ax.hist(scores[y_binary == lbl], bins=100, alpha=0.55,
                label=name, color=color, density=True)
    ax.set_xlabel("LDA Score")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_multilabel(embeddings, labels, lda, path):
    """2-D scatter of first two LDA components."""
    proj = lda.transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, name in LABEL_NAMES.items():
        mask = labels == lbl
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   alpha=0.25, s=8, label=name, color=COLORS[lbl])
    ax.set_xlabel("LDA Component 1")
    ax.set_ylabel("LDA Component 2")
    plt.xlim(-12,5)
    ax.legend(markerscale=4, fontsize=13, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=len(LABEL_NAMES) // 2 + 1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def fit_and_evaluate(embeddings, labels, label_names, n_components=None):
    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_tr, y_tr)
    y_pred = lda.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    target_names = [label_names[i] for i in sorted(label_names)]
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=target_names))
    return lda, X_te, y_te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = np.load(EMB_PATH)
    embeddings = data["embeddings"]
    labels     = data["labels"]

    print(f"Loaded: {embeddings.shape[0]} sentences, {embeddings.shape[1]}-dim embeddings")
    for lbl, name in LABEL_NAMES.items():
        print(f"  label {lbl} ({name}): {(labels == lbl).sum()}")

    # --- Binary LDA ---
    # print("\n=== Binary LDA: Human vs. AI ===")
    # y_binary = (labels != 0).astype(int)
    # binary_label_names = {0: "Human", 1: "AI (all)"}
    # lda_bin, X_te_bin, y_te_bin = fit_and_evaluate(embeddings, y_binary, binary_label_names, n_components=1)
    # plot_binary(
    #     X_te_bin, y_te_bin, lda_bin,
    #     os.path.join(OUT_DIR, "lda_binary.pdf")
    # )

    # --- Multi-Label LDA ---
    print("\n=== Multi-Label LDA: Human vs. Individual LLMs ===")
    lda_multi, X_te_multi, y_te_multi = fit_and_evaluate(embeddings, labels, LABEL_NAMES, n_components=4)
    plot_multilabel(
        X_te_multi, y_te_multi, lda_multi,
        os.path.join(OUT_DIR, "lda_multilabel.pdf")
    )


if __name__ == "__main__":
    main()
