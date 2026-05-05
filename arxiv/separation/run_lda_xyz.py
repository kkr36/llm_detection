"""
Run Linear Discriminant Analysis on XYZ rewrite sentence embeddings.

Four tasks:
  Binary      — Human (0) vs. all AI (labels 1–6)
  Multi-label — all 7 classes (human + 6 rewrites)
  PN subset   — {human_abstract, rewrite_X, rewrite_Z, rewrite_Z_1_PN, rewrite_Z_2_PN}
  PU subset   — {human_abstract, rewrite_X, rewrite_Z, rewrite_Z_1_PU, rewrite_Z_2_PU}

Label map:
  0 = human_abstract
  1 = rewrite_X
  2 = rewrite_Z
  3 = rewrite_Z_1_PU
  4 = rewrite_Z_1_PN
  5 = rewrite_Z_2_PU
  6 = rewrite_Z_2_PN

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

plt.rcParams.update({"font.size": 14})

EMB_PATH = "/home/kkr36/llm_detection/arxiv/separation/embeddings_xyz.npz"
OUT_DIR  = "/home/kkr36/llm_detection/arxiv/separation"

LABEL_NAMES = {
    # 0: "Human",
    1: "rewrite_X",
    2: "rewrite_Z",
    3: "rewrite_Z_1_PU",
    4: "rewrite_Z_1_PN",
    5: "rewrite_Z_2_PU",
    6: "rewrite_Z_2_PN",
}

COLORS = {
    # 0: "#2196F3",  # blue
    1: "#F44336",  # red
    2: "#4CAF50",  # green
    3: "#FF9800",  # orange
    4: "#9C27B0",  # purple
    5: "#00BCD4",  # cyan
    6: "#795548",  # brown
}

# Subset definitions
PN_LABELS = {
    # 0: "Human",
    1: "rewrite_X", 2: "rewrite_Z", 4: "rewrite_Z_1_PN", 6: "rewrite_Z_2_PN"}
PU_LABELS = {
    # 0: "Human",
    1: "rewrite_X", 2: "rewrite_Z", 3: "rewrite_Z_1_PU", 5: "rewrite_Z_2_PU"}

HUMAN_X_Z2PU_LABELS = {0: "Human", 1: "rewrite_X", 5: "rewrite_Z_2_PU"}
HUMAN_X_Z2PN_LABELS = {0: "Human", 1: "rewrite_X", 6: "rewrite_Z_2_PN"}
HUMAN_COLORS = {**COLORS, 0: "#2196F3"}


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


def plot_multilabel(embeddings, labels, label_names, lda, title, path, colors=None):
    """2-D scatter of first two LDA components."""
    if colors is None:
        colors = COLORS
    proj = lda.transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, name in label_names.items():
        mask = labels == lbl
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   alpha=0.25, s=8, label=name, color=colors[lbl])
    ax.set_xlabel("LDA Component 1")
    ax.set_ylabel("LDA Component 2")
    ax.legend(markerscale=4, fontsize=13)
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
    sorted_lbls = sorted(label_names)
    target_names = [label_names[i] for i in sorted_lbls]
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, labels=sorted_lbls, target_names=target_names))
    return lda


def subset(embeddings, labels, keep_labels):
    """Filter embeddings/labels to only the specified label set, remapping labels."""
    mask = np.isin(labels, list(keep_labels))
    return embeddings[mask], labels[mask]


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
    # lda_bin = fit_and_evaluate(embeddings, y_binary, binary_label_names, n_components=1)
    # plot_binary(
    #     embeddings, y_binary, lda_bin,
    #     os.path.join(OUT_DIR, "lda_binary_xyz.png")
    # )

    # --- Full Multi-Label LDA ---
    # print("\n=== Multi-Label LDA: All Classes ===")
    # n_classes = len(LABEL_NAMES)
    # lda_multi = fit_and_evaluate(embeddings, labels, LABEL_NAMES, n_components=n_classes - 1)
    # plot_multilabel(
    #     embeddings, labels, LABEL_NAMES, lda_multi,
    #     "Multi-Label LDA: Human vs. XYZ Rewrites (All)",
    #     os.path.join(OUT_DIR, "lda_multilabel_xyz.png")
    # )

    # --- PN Subset LDA ---
    print("\n=== Multi-Label LDA: PN Subset ===")
    emb_pn, lbl_pn = subset(embeddings, labels, PN_LABELS)
    n_pn = len(PN_LABELS)
    lda_pn = fit_and_evaluate(emb_pn, lbl_pn, PN_LABELS, n_components=n_pn - 1)
    plot_multilabel(
        emb_pn, lbl_pn, PN_LABELS, lda_pn,
        "Multi-Label LDA: Human / rewrite_X / rewrite_Z / PN variants",
        os.path.join(OUT_DIR, "lda_multilabel_xyz_PN.pdf")
    )

    # --- PU Subset LDA ---
    print("\n=== Multi-Label LDA: PU Subset ===")
    emb_pu, lbl_pu = subset(embeddings, labels, PU_LABELS)
    n_pu = len(PU_LABELS)
    lda_pu = fit_and_evaluate(emb_pu, lbl_pu, PU_LABELS, n_components=n_pu - 1)
    plot_multilabel(
        emb_pu, lbl_pu, PU_LABELS, lda_pu,
        "Multi-Label LDA: Human / rewrite_X / rewrite_Z / PU variants",
        os.path.join(OUT_DIR, "lda_multilabel_xyz_PU.pdf")
    )

    # --- Human / rewrite_X / rewrite_Z_2_PU ---
    print("\n=== Multi-Label LDA: Human + rewrite_X + rewrite_Z_2_PU ===")
    emb_h_z2pu, lbl_h_z2pu = subset(embeddings, labels, HUMAN_X_Z2PU_LABELS)
    lda_h_z2pu = fit_and_evaluate(emb_h_z2pu, lbl_h_z2pu, HUMAN_X_Z2PU_LABELS, n_components=2)
    plot_multilabel(
        emb_h_z2pu, lbl_h_z2pu, HUMAN_X_Z2PU_LABELS, lda_h_z2pu,
        "LDA: Human / rewrite_X / rewrite_Z_2_PU",
        os.path.join(OUT_DIR, "lda_human_X_Z2PU.pdf"),
        colors=HUMAN_COLORS,
    )

    # --- Human / rewrite_X / rewrite_Z_2_PN ---
    print("\n=== Multi-Label LDA: Human + rewrite_X + rewrite_Z_2_PN ===")
    emb_h_z2pn, lbl_h_z2pn = subset(embeddings, labels, HUMAN_X_Z2PN_LABELS)
    lda_h_z2pn = fit_and_evaluate(emb_h_z2pn, lbl_h_z2pn, HUMAN_X_Z2PN_LABELS, n_components=2)
    plot_multilabel(
        emb_h_z2pn, lbl_h_z2pn, HUMAN_X_Z2PN_LABELS, lda_h_z2pn,
        "LDA: Human / rewrite_X / rewrite_Z_2_PN",
        os.path.join(OUT_DIR, "lda_human_X_Z2PN.pdf"),
        colors=HUMAN_COLORS,
    )


if __name__ == "__main__":
    main()
