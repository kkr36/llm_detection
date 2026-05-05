#!/home/kkr36/.conda/envs/llm_embeddings/bin/python
"""
Temporal LDA: how human/AI embedding distributions shift across years 2010–2020.

Plots (n=4 AI sources, total n+3=8 files):
  1. lda_years_binary_2d.png     — Ridgeline: binary LDA score dist per year (human vs AI)
  2. lda_years_binary_hist.png   — Histogram: binary LDA scores, human vs all AI (pooled)
  3. lda_years_overview.png      — 2×3 grid: multi-class LDA 2D scatter, one panel per year
  4–7. lda_years_{llm}.png       — Per-LLM: 2D scatter (year-colored) + per-year density curves
  8. lda_years_trajectories.png  — Centroid trajectories: year×class LDA, arrows show time
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

plt.rcParams.update({"font.size": 14})

EMB_PATH = "/home/kkr36/llm_detection/arxiv/separation/embeddings_years.npz"
OUT_DIR  = "/home/kkr36/llm_detection/arxiv/separation"
YEARS    = [2010, 2012, 2014, 2016, 2018, 2020]

LABEL_NAMES = {
    0: "Human",
    1: "GPT OSS 120b",
    2: "Llama 3.3 70b",
    3: "Gemini 3 Preview",
    4: "Qwen",
}
AI_LABELS = [1, 2, 3, 4]
LLM_SLUGS = {1: "gpt", 2: "llama", 3: "gemini", 4: "qwen"}

CLASS_COLORS = {
    0: "#2196F3",  # blue  – human
    1: "#F44336",  # red   – GPT
    2: "#4CAF50",  # green – Llama
    3: "#FF9800",  # orange– Gemini
    4: "#9C27B0",  # purple– Qwen
}

_yr_positions = np.linspace(0.1, 0.9, len(YEARS))
YEAR_COLORS = {yr: plt.cm.plasma(_yr_positions[i]) for i, yr in enumerate(YEARS)}

SEED       = 42
SCATTER_N  = 300   # max points per (class, year) in scatter subplots
KDE_BW     = 0.25
HIST_BINS  = 60


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    data = np.load(EMB_PATH)
    embs, lbls, yrs = [], [], []
    for year in YEARS:
        e = data[f"embeddings_{year}"]
        l = data[f"labels_{year}"]
        embs.append(e)
        lbls.append(l)
        yrs.append(np.full(len(l), year, dtype=np.int32))
    return np.vstack(embs), np.concatenate(lbls), np.concatenate(yrs)


# ---------------------------------------------------------------------------
# LDA fitting
# ---------------------------------------------------------------------------

def fit_and_report(emb, lbl, label_names, n_components=None, tag=""):
    X_tr, X_te, y_tr, y_te = train_test_split(
        emb, lbl, test_size=0.2, random_state=SEED, stratify=lbl
    )
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_tr, y_tr)
    y_pred = lda.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    sorted_lbls = sorted(label_names)
    tag_str = f" [{tag}]" if tag else ""
    print(f"\n=== LDA{tag_str} — Accuracy: {acc:.4f} ===")
    print(classification_report(
        y_te, y_pred,
        labels=sorted_lbls,
        target_names=[label_names[l] for l in sorted_lbls],
    ))
    # Refit on full data for projection
    lda_full = LinearDiscriminantAnalysis(n_components=n_components)
    lda_full.fit(emb, lbl)
    return lda_full


# ---------------------------------------------------------------------------
# Plot 1: Binary ridgeline — human vs AI per year
# ---------------------------------------------------------------------------

def plot_binary_2d(emb, lbl, yr, lda_bin, path):
    scores = lda_bin.transform(emb).flatten()
    y_bin  = (lbl != 0).astype(int)

    x_lo = scores.min() - 0.3
    x_hi = scores.max() + 0.3
    x_range = np.linspace(x_lo, x_hi, 500)
    gap = 1.6

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, year in enumerate(YEARS):
        base = i * gap
        for is_ai, (color, label) in enumerate([
            (CLASS_COLORS[0], "Human"),
            ("#EF5350", "AI (all)"),
        ]):
            mask = (yr == year) & (y_bin == is_ai)
            s = scores[mask]
            if len(s) < 5:
                continue
            try:
                kde = gaussian_kde(s, bw_method=KDE_BW)
                density = kde(x_range)
                density = density / density.max() * (gap * 0.85)
            except Exception:
                continue
            ax.fill_between(x_range, base, base + density, alpha=0.45, color=color)
            ax.plot(x_range, base + density, color=color, linewidth=0.9, alpha=0.85)

        ax.axhline(base, color="gray", linewidth=0.4, alpha=0.4)
        ax.text(x_hi + 0.15, base + gap * 0.28, str(year), fontsize=13, va="center")

    ax.legend(handles=[
        Patch(color=CLASS_COLORS[0], alpha=0.6, label="Human"),
        Patch(color="#EF5350",        alpha=0.6, label="AI (all)"),
    ], loc="upper left", fontsize=14)

    ax.set_xlabel("Binary LDA Score", fontsize=15)
    ax.set_xlim(x_lo, x_hi + 0.8)
    ax.set_yticks([])
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Binary histogram (all years pooled)
# ---------------------------------------------------------------------------

def plot_binary_hist(emb, lbl, lda_bin, path):
    scores = lda_bin.transform(emb).flatten()
    y_bin  = (lbl != 0).astype(int)

    fig, ax = plt.subplots(figsize=(9, 4))
    for is_ai, (color, label) in enumerate([
        (CLASS_COLORS[0], "Human"),
        ("#EF5350", "AI (all)"),
    ]):
        ax.hist(scores[y_bin == is_ai], bins=HIST_BINS, alpha=0.55,
                density=True, color=color, label=label)

    ax.set_xlabel("Binary LDA Score", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Overview — multi-class LDA 2D, one subplot per year
# ---------------------------------------------------------------------------

def plot_overview(emb, lbl, yr, lda_multi, path):
    proj = lda_multi.transform(emb)
    rng  = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)

    for ax, year in zip(axes.flat, YEARS):
        mask_yr = yr == year
        for label, name in LABEL_NAMES.items():
            mask = mask_yr & (lbl == label)
            idx = np.where(mask)[0]
            idx = rng.choice(idx, size=min(SCATTER_N, len(idx)), replace=False)
            ax.scatter(proj[idx, 0], proj[idx, 1],
                       alpha=0.3, s=8, color=CLASS_COLORS[label], label=name)
        ax.set_xlabel("LDA 1")
        ax.set_ylabel("LDA 2")

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CLASS_COLORS[l], markersize=9, label=n)
        for l, n in LABEL_NAMES.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=13, markerscale=1.5, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plots 4-7: Per-LLM panels (2D scatter + per-year density curves)
# ---------------------------------------------------------------------------

def plot_per_llm(emb, lbl, yr, lda_multi, ai_label, path):
    name     = LABEL_NAMES[ai_label]
    ai_color = CLASS_COLORS[ai_label]

    # Subset to human + this LLM
    mask     = (lbl == 0) | (lbl == ai_label)
    emb_sub  = emb[mask]
    lbl_sub  = lbl[mask]
    yr_sub   = yr[mask]
    y_bin    = (lbl_sub != 0).astype(int)

    # Binary LDA for density plot
    lda_bin = fit_and_report(
        emb_sub, y_bin,
        {0: "Human", 1: name},
        n_components=1,
        tag=f"Human vs {name}",
    )
    scores = lda_bin.transform(emb_sub).flatten()

    # Multi-class 2D projection (axes calibrated on full 5-class space)
    proj = lda_multi.transform(emb_sub)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left: 2D scatter in multi-class LDA space, colored by year ---
    rng = np.random.default_rng(SEED)
    for year in YEARS:
        yc = YEAR_COLORS[year]
        for lbl_val, marker in [(0, "o"), (ai_label, "^")]:
            mask2 = (lbl_sub == lbl_val) & (yr_sub == year)
            idx   = np.where(mask2)[0]
            idx   = rng.choice(idx, size=min(SCATTER_N, len(idx)), replace=False)
            ax_left.scatter(proj[idx, 0], proj[idx, 1],
                            alpha=0.35, s=10, color=yc, marker=marker)

    year_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=YEAR_COLORS[yr], markersize=8, label=str(yr))
        for yr in YEARS
    ]
    class_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="Human"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=8, label=name),
    ]
    leg1 = ax_left.legend(handles=year_handles, title="Year",
                           loc="upper left", fontsize=11, title_fontsize=12)
    ax_left.add_artist(leg1)
    ax_left.legend(handles=class_handles, loc="lower right", fontsize=13)
    ax_left.set_xlabel("LDA Component 1", fontsize=14)
    ax_left.set_ylabel("LDA Component 2", fontsize=14)

    # --- Right: per-year KDE density curves (binary LDA scores) ---
    x_range = np.linspace(scores.min() - 0.1, scores.max() + 0.1, 400)
    for year in YEARS:
        yc = YEAR_COLORS[year]
        for is_ai, ls in [(0, "--"), (1, "-")]:
            mask2 = (yr_sub == year) & (y_bin == is_ai)
            s = scores[mask2]
            if len(s) < 5:
                continue
            try:
                density = gaussian_kde(s, bw_method=KDE_BW)(x_range)
                ax_right.plot(x_range, density, color=yc, linestyle=ls,
                              linewidth=1.5, alpha=0.85)
            except Exception:
                pass

    hist_handles = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="Human"),
        Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.5, label=name),
    ] + [
        Line2D([0], [0], color=YEAR_COLORS[yr], linewidth=2.5, label=str(yr))
        for yr in YEARS
    ]
    ax_right.legend(handles=hist_handles, fontsize=12, ncol=2)
    ax_right.set_xlabel("Binary LDA Score", fontsize=14)
    ax_right.set_ylabel("Density", fontsize=14)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 8: Centroid trajectories — year×class LDA
# ---------------------------------------------------------------------------

def plot_centroid_trajectories(emb, lbl, yr, path):
    # Composite label: unique per (year, class). year*10+lbl works since lbl is 0–4.
    composite = yr * 10 + lbl
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(emb, composite)
    proj = lda.transform(emb)

    fig, ax = plt.subplots(figsize=(11, 8))

    for label, name in LABEL_NAMES.items():
        color = CLASS_COLORS[label]
        centroids = []
        for year in YEARS:
            mask = (lbl == label) & (yr == year)
            if mask.sum() == 0:
                continue
            centroids.append((year, proj[mask, 0].mean(), proj[mask, 1].mean()))

        if not centroids:
            continue

        years_plot, xs, ys = zip(*centroids)
        ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.4, zorder=1)
        ax.scatter(xs, ys, color=color, s=70, zorder=3)

        for i in range(len(centroids) - 1):
            ax.annotate(
                "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=1.8,
                    shrinkA=6, shrinkB=6,
                ),
                zorder=2,
            )

        for year, x, y in centroids:
            ax.annotate(
                str(year), (x, y),
                textcoords="offset points", xytext=(5, 4),
                fontsize=11, color=color, alpha=0.9,
            )

    handles = [
        Line2D([0], [0], color=CLASS_COLORS[l], linewidth=2,
               marker="o", markersize=7, label=n)
        for l, n in LABEL_NAMES.items()
    ]
    ax.legend(handles=handles, fontsize=13, loc="best")
    ax.set_xlabel("LDA Component 1 (year×class LDA)", fontsize=15)
    ax.set_ylabel("LDA Component 2 (year×class LDA)", fontsize=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading embeddings...")
    emb, lbl, yr = load_data()
    print(f"Total: {len(emb)} sentences, {emb.shape[1]}-dim embeddings")
    for year in YEARS:
        print(f"  {year}: {(yr == year).sum()} sentences")

    # Binary LDA on all years combined
    y_binary = (lbl != 0).astype(int)
    lda_bin = fit_and_report(
        emb, y_binary, {0: "Human", 1: "AI (all)"},
        n_components=1, tag="Binary: Human vs All AI"
    )

    # Multi-class LDA on all years combined (used for 2D projections)
    lda_multi = fit_and_report(
        emb, lbl, LABEL_NAMES,
        n_components=2, tag="Multi-class: All Sources"
    )

    print("\nGenerating plots...")

    # Plot 1: binary ridgeline (2D temporal view)
    plot_binary_2d(emb, lbl, yr, lda_bin,
                   os.path.join(OUT_DIR, "lda_years_binary_2d.pdf"))

    # Plot 2: binary histogram (all years pooled)
    plot_binary_hist(emb, lbl, lda_bin,
                     os.path.join(OUT_DIR, "lda_years_binary_hist.pdf"))

    # Plot 3: overview 2×3 grid (multi-class LDA per year)
    plot_overview(emb, lbl, yr, lda_multi,
                  os.path.join(OUT_DIR, "lda_years_overview.pdf"))

    # Plots 4-7: per-LLM panels
    for ai_label in AI_LABELS:
        slug = LLM_SLUGS[ai_label]
        plot_per_llm(emb, lbl, yr, lda_multi, ai_label,
                     os.path.join(OUT_DIR, f"lda_years_{slug}.pdf"))

    # Plot 8: centroid trajectories (year×class LDA)
    plot_centroid_trajectories(emb, lbl, yr,
                               os.path.join(OUT_DIR, "lda_years_trajectories.pdf"))


if __name__ == "__main__":
    main()
