"""
Plot predicted %% AI in the held-out test set vs. year, one connected line per
detection method (Pangram / PU / James).

Reads whatever exists in results/:
    james_mpe.csv    -> James MPE estimate                (metric: mpe)
    pu_mpe.csv       -> PU BBE estimate + avg P(AI)        (metric: mpe | mean_ai_prob)
    pangram_mpe.csv  -> Pangram continuous | binary label  (metric: mean_fraction_ai | frac_pred_ai)
Missing methods are simply skipped, so this can be run with James+PU now and
re-run after Pangram is available.

    cd .../PU_learning && python mpe_year/plot_mpe_year.py
    # options: --pu-metric {mpe,mean_ai_prob}  --pangram-metric {mean_fraction_ai,frac_pred_ai}
    #          --overlay-alt  (also draw the alternate PU/Pangram metric, dashed, same hue)

Static matplotlib figure (PDF+PNG), consistent with the repo's other figure_*.py.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match the style of plot_helper/plot_temporal_alpha.py (temporal_alpha_grid.pdf):
# global bold font at size 20.
matplotlib.rc("font", **{"weight": "bold", "size": 20})

RESULTS = os.path.join(os.path.dirname(__file__), "results")
YEARS = [2020, 2023, 2025]

# Okabe-Ito colorblind-safe hues, assigned in fixed order (identity follows the
# method, never its rank). One hue per method.
COLORS = {
    "pangram": "#0072B2",  # blue
    "pu":      "#D55E00",  # vermillion
    "james":   "#009E73",  # green
}
LABELS = {
    "pangram": "Pangram",
    "pu":      "PU (avg P(AI))",
    "james":   "James (MPE)",
}


def _read(name):
    path = os.path.join(RESULTS, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def series_for(method, pu_metric, pangram_metric):
    """Return (years, pct, alt_pct_or_None) for a method, or None if unavailable."""
    if method == "james":
        df = _read("james_mpe.csv")
        col, alt = "mpe", None
    elif method == "pu":
        df = _read("pu_mpe.csv")
        col = pu_metric
        alt = "mean_ai_prob" if pu_metric == "mpe" else "mpe"
    elif method == "pangram":
        df = _read("pangram_mpe.csv")
        col = pangram_metric
        alt = "frac_pred_ai" if pangram_metric == "mean_fraction_ai" else "mean_fraction_ai"
    else:
        return None
    if df is None or col not in df.columns:
        return None
    df = df.set_index("year")
    years = [y for y in YEARS if y in df.index]
    pct = [100.0 * float(df.loc[y, col]) for y in years]
    alt_pct = None
    if alt is not None and alt in df.columns:
        alt_pct = [100.0 * float(df.loc[y, alt]) for y in years]
    return years, pct, alt_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pu-metric", choices=["mpe", "mean_ai_prob"], default="mean_ai_prob")
    ap.add_argument("--pangram-metric", choices=["mean_fraction_ai", "frac_pred_ai"],
                    default="mean_fraction_ai")
    ap.add_argument("--overlay-alt", action="store_true",
                    help="also draw the alternate PU/Pangram metric (dashed, same hue)")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plotted = []
    for method in ("pangram", "pu", "james"):  # fixed categorical order
        got = series_for(method, args.pu_metric, args.pangram_metric)
        if got is None:
            continue
        years, pct, alt_pct = got
        c = COLORS[method]
        ax.plot(years, pct, marker="o", markersize=8, linewidth=2, color=c,
                label=LABELS[method], zorder=3, clip_on=False)
        # direct label at the last point (nudge Pangram's down to clear PU's label)
        y_off = -14 if method == "pangram" else 4
        ax.annotate(f"{pct[-1]:.1f}%", (years[-1], pct[-1]),
                    textcoords="offset points", xytext=(8, y_off), color=c,
                    fontsize=16, fontweight="bold")
        if args.overlay_alt and alt_pct is not None:
            ax.plot(years, alt_pct, marker="o", markersize=5, linewidth=1.5,
                    linestyle="--", color=c, alpha=0.55, zorder=2, clip_on=False)
        plotted.append(method)

    if not plotted:
        raise SystemExit("No results found in results/. Run the method scripts first.")

    ax.set_xticks(YEARS)
    ax.set_xlabel("Test Year")
    ax.set_ylabel("Predicted % AI")
    top = max(ax.get_ylim()[1], 1.0)
    ax.set_ylim(bottom=0, top=top * 1.12)  # headroom so last-point labels don't clip
    ax.margins(x=0.06)
    if args.overlay_alt and "pu" in plotted:
        ax.plot([], [], linestyle="--", color=COLORS["pu"], alpha=0.7, linewidth=1.5,
                label="PU (avg P(AI))")
    # Horizontal, frameless legend across the top, like temporal_alpha_grid.pdf
    ncol = len(plotted) + (1 if args.overlay_alt and "pu" in plotted else 0)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=ncol, columnspacing=1.0, handletextpad=0.5)

    fig.tight_layout()
    out_pdf = os.path.join(RESULTS, "mpe_year_plot.pdf")
    out_png = os.path.join(RESULTS, "mpe_year_plot.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"plotted methods: {plotted}")
    print(f"saved -> {out_pdf}\n         {out_png}")


if __name__ == "__main__":
    main()
