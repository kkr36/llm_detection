"""
Rank abstracts in results_4_25.csv by % n-gram overlap between the
human abstract and its mirror_4 rewrite.

Adapted from ../analyze_ngram_overlap.py.
"""

import ast
import re
import os
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_FILE  = "results_4_25.csv"
MIRROR_COL = "mirror_4"
N_RANGE   = list(range(1, 16))   # 1..15
SORT_N    = 3                     # n-gram size used for the ranking table
PLOT_FILE = "ngram_overlap_ranked.png"
OUT_CSV   = "ngram_overlap_ranked.csv"
# ───────────────────────────────────────────────────────────────────────────────


def parse_text(cell) -> str:
    if pd.isna(cell):
        return ""
    cell = str(cell).strip()
    try:
        val = ast.literal_eval(cell)
        if isinstance(val, list):
            return " ".join(str(s) for s in val)
        return str(val)
    except Exception:
        return cell


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def overlap_fraction(human_tokens, mirror_tokens, n: int) -> float:
    """Fraction of human n-grams found verbatim in mirror n-grams."""
    h_set = set(ngrams(human_tokens, n))
    m_grams = ngrams(mirror_tokens, n)
    total = len(h_set)
    if total == 0:
        return float("nan")
    count = sum(1 for g in m_grams if g in h_set)
    return count / total


def main():
    df = pd.read_csv(CSV_FILE)

    records = []
    for row_idx, row in df.iterrows():
        human_tokens  = tokenize(parse_text(row["human"]))
        mirror_tokens = tokenize(parse_text(row[MIRROR_COL]))
        row_rec = {"row_idx": row_idx, "abstract": parse_text(row["human"]).replace("\n", " ").strip()}
        for n in N_RANGE:
            row_rec[f"overlap_n{n}"] = overlap_fraction(human_tokens, mirror_tokens, n)
        records.append(row_rec)

    result = pd.DataFrame(records)
    result = result.sort_values(f"overlap_n{SORT_N}", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)

    result.to_csv(OUT_CSV, index=False)
    print(f"Saved ranked CSV → {OUT_CSV}\n")

    # ── Print table ─────────────────────────────────────────────────────────────
    col = f"overlap_n{SORT_N}"
    print(f"{'Rank':<5} {'Overlap% (n='+str(SORT_N)+')':<22} {'Abstract (first 100 chars)'}")
    print("-" * 130)
    for _, row in result.iterrows():
        pct = row[col] * 100
        preview = row["abstract"][:100]
        print(f"{int(row['rank']):<5} {pct:<22.2f} {preview}")

    # ── Plot: overlap% vs n for each abstract, coloured by rank ─────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.cm.RdYlGn_r
    n_rows = len(result)
    for i, (_, row) in enumerate(result.iterrows()):
        ys = [row[f"overlap_n{n}"] * 100 for n in N_RANGE]
        color = cmap(i / max(n_rows - 1, 1))
        label = f"#{int(row['rank'])}" if int(row['rank']) <= 5 else None
        ax.plot(N_RANGE, ys, color=color, alpha=0.6, linewidth=1.2, label=label)

    ax.set_xlabel("N (n-gram size)", fontsize=12)
    ax.set_ylabel("% human n-grams found in mirror", fontsize=11)
    ax.set_title(f"N-gram overlap by abstract — {CSV_FILE}  ·  {MIRROR_COL}\n"
                 f"(sorted by n={SORT_N}; red=highest overlap, green=lowest)", fontsize=12)
    ax.set_xticks(N_RANGE)
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend(title="Top-5 ranks", fontsize=9)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label="rank (red=1st)")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot → {PLOT_FILE}")


if __name__ == "__main__":
    main()
