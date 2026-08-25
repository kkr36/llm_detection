"""
Fast-DetectGPT robustness-across-generators figure.

Reads the two AUC logs
    ../logging_accuracy_llm_fastdetect_sentence.csv
    ../logging_accuracy_llm_fastdetect_abstract.csv
and renders two annotated AUC heat-tables (one per granularity): rows = the
Fast-DetectGPT reference-model configuration (single-model surrogate, or the
two-model sampling->scoring pair), columns = the test LLM whose text is being
detected. Cells are AUC.

The story is robustness: a robust zero-shot detector would be uniformly high
across generators. Instead AUC swings from ~0.99 (Llama) down THROUGH 0.5
(useless) to ~0.25 (Codex, anti-correlated -- its text looks *more* human than
real humans to the detector). So AUC is a POLARITY quantity pivoting at 0.5, and
the color is a diverging map centered there: blue = works, gray = chance (0.5),
red = worse than chance. Every cell is also annotated, so identity never rests on
color alone.

AUC is invariant to the score->prob variant (raw/platt/oracle all share the
canonical -d ranking), so we read the FastDetectGPT-raw rows only.

Usage (env with pandas + matplotlib, e.g. llm_embeddings):
    python plot_helper/plot_fastdetect_robustness.py
Writes plot_helper/fastdetect_robustness.{pdf,png}.
"""
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)
VARIANT = "FastDetectGPT-raw"   # AUC identical across variants

# ---- validated diverging palette (dataviz skill reference instance) ----
POLE_LOW = "#d03b3b"    # red  -> low AUC / anti-correlated / detector fails
MID = "#f0efec"         # neutral gray midpoint == chance (0.5)
POLE_HIGH = "#2a78d6"   # blue -> high AUC / detector works
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def diverging_at(lo, hi, center=0.5):
    """A red->gray->blue map on a plain [lo,hi] Normalize, with the gray midpoint
    pinned to `center` (=chance, 0.5). Plain Normalize keeps the colorbar well-behaved
    (TwoSlopeNorm's nonlinear axis breaks colorbar tick placement)."""
    frac = (center - lo) / (hi - lo)
    cmap = LinearSegmentedColormap.from_list(
        "auc_div", [(0.0, POLE_LOW), (frac, MID), (1.0, POLE_HIGH)])
    return cmap, Normalize(vmin=lo, vmax=hi)

# short, readable labels for the reference-model configs
REF_LABEL = {
    "EleutherAI/gpt-neo-2.7B": "gpt-neo-2.7B\n(single-model)",
    "EleutherAI/gpt-j-6B -> EleutherAI/gpt-neo-2.7B": "gpt-j-6B → gpt-neo-2.7B\n(two-model pair)",
    "Qwen/Qwen2.5-7B": "Qwen2.5-7B\n(single-model)",
}
# render rows in this order when present (pair on top: the paper's headline config)
REF_ORDER = [
    "EleutherAI/gpt-j-6B -> EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neo-2.7B",
    "Qwen/Qwen2.5-7B",
]
# compact column labels so five generators fit without colliding
LLM_LABEL = {
    "Llama 3.3 70b Instruct": "Llama 3.3 70B",
    "GPT OSS 120b": "GPT-OSS 120B",
    "Gemini 3 Preview": "Gemini 3",
    "Qwen": "Qwen",
    "Codex": "Codex",
}


def load_matrix(granularity):
    df = pd.read_csv(os.path.join(DATA_DIR, f"logging_accuracy_llm_fastdetect_{granularity}.csv"))
    df = df[df["learning_method"] == VARIANT]
    piv = df.pivot_table(index="ref_model", columns="test_llm", values="auc")
    rows = [r for r in REF_ORDER if r in piv.index] + [r for r in piv.index if r not in REF_ORDER]
    return piv.reindex(rows)


def _text_color(rgba):
    """Black on light cells, white on saturated poles (WCAG relative luminance)."""
    r, g, b = rgba[:3]
    lin = [(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.2216 * lin[2]
    return "#ffffff" if lum < 0.45 else INK


def draw_panel(ax, mat, cmap, norm, title, show_xlabels=True):
    n_rows, n_cols = mat.shape
    # pcolormesh with a surface-colored edge gives the 2px inter-cell gap the spec wants.
    ax.set_facecolor(SURFACE)
    x = np.arange(n_cols + 1)
    y = np.arange(n_rows + 1)
    mesh = ax.pcolormesh(x, y, mat.values, cmap=cmap, norm=norm,
                         edgecolors=SURFACE, linewidth=3)
    ax.invert_yaxis()  # first row on top

    for i in range(n_rows):
        for j in range(n_cols):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="medium",
                    color=_text_color(cmap(norm(v))))

    ax.set_xticks(np.arange(n_cols) + 0.5)
    if show_xlabels:
        ax.set_xticklabels([LLM_LABEL.get(c, c) for c in mat.columns], fontsize=11, color=INK)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([REF_LABEL.get(r, r) for r in mat.index], fontsize=10.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=10, loc="left")
    ax.set_aspect("auto")
    return mesh


def main():
    sent = load_matrix("sentence")
    abst = load_matrix("abstract")

    # shared column order: by two-model-pair abstract AUC, descending -> robustness gradient
    pair = "EleutherAI/gpt-j-6B -> EleutherAI/gpt-neo-2.7B"
    order = abst.loc[pair].sort_values(ascending=False).index.tolist()
    sent = sent.reindex(columns=order)
    abst = abst.reindex(columns=order)

    # one diverging colormap+norm across both panels, gray pinned at 0.5 == chance
    lo = float(min(np.nanmin(sent.values), np.nanmin(abst.values)))
    hi = float(max(np.nanmax(sent.values), np.nanmax(abst.values)))
    cmap, norm = diverging_at(lo, hi, center=0.5)

    fig = plt.figure(figsize=(9.2, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[sent.shape[0], abst.shape[0]],
                          hspace=0.30, left=0.26, right=0.97, top=0.85, bottom=0.20)
    ax_s = fig.add_subplot(gs[0]); ax_a = fig.add_subplot(gs[1])
    draw_panel(ax_s, sent, cmap, norm, "Sentence level", show_xlabels=False)
    mesh = draw_panel(ax_a, abst, cmap, norm, "Abstract level", show_xlabels=True)

    fig.suptitle("Fast-DetectGPT AUC by test LLM — robustness across generators",
                 fontsize=15, fontweight="bold", color=INK, x=0.26, ha="left", y=0.95)

    # shared horizontal colorbar spanning both panels; use the mesh itself as the mappable
    # (a bare ScalarMappable + TwoSlopeNorm renders empty and warns), single-line ticks.
    cb = fig.colorbar(mesh, ax=[ax_s, ax_a], orientation="horizontal",
                      fraction=0.06, pad=0.14, aspect=42)
    cb.set_ticks([round(lo, 2), 0.5, round(hi, 2)])
    cb.set_ticklabels([f"{lo:.2f}", "0.50", f"{hi:.2f}"])
    cb.ax.tick_params(labelsize=10, length=0, colors=INK_MUTED)
    cb.set_label("AUROC — 0.50 = chance; below 0.50, rewrites look more human than real humans",
                 fontsize=9.5, color=INK_MUTED, labelpad=8)
    cb.outline.set_visible(False)
    # pole words at the two ends, in the pole hues (identity beyond color alone)
    cb.ax.text(-0.012, 0.5, "anti-correlated", transform=cb.ax.transAxes,
               fontsize=9, color=POLE_LOW, ha="right", va="center")
    cb.ax.text(1.012, 0.5, "reliable", transform=cb.ax.transAxes,
               fontsize=9, color=POLE_HIGH, ha="left", va="center")

    for ext in ("pdf", "png"):
        out = os.path.join(HERE, f"fastdetect_robustness.{ext}")
        fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
