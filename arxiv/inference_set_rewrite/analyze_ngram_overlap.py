"""
Analyze per-row n-gram verbatim overlap between mirror_t columns and the human column
across all results_*.csv files. Outputs a summary CSV and per-file boxplot figures.
"""

import ast
import re
import glob
import os
import sys
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
N_RANGE = list(range(1, 25))  # 1..15
CSV_GLOBS = [
    "iterative_prompt_rewrite_test/results_*.csv",
    "iterative_prompt_rewrite_test_pangram/results_*.csv",
    "old_csvs/results_*.csv",
]
PLOT_DIR = "ngram_overlap_plots"
SUMMARY_CSV = "ngram_overlap_summary.csv"
# ───────────────────────────────────────────────────────────────────────────────


def parse_text(cell) -> str:
    """Parse a cell that is either a Python list-of-strings or a plain string."""
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
    """Lowercase and strip punctuation, return word tokens."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def overlap_fraction(human_tokens, mirror_tokens, n: int) -> dict:
    """Fraction of mirror n-grams that appear verbatim in human n-grams."""
    h_set = set(ngrams(human_tokens, n))
    m_grams = ngrams(mirror_tokens, n)
    total = len(m_grams)
    if total == 0:
        return {"overlap_fraction": float("nan"), "overlap_count": 0, "mirror_ngram_total": 0}
    count = sum(1 for g in m_grams if g in h_set)
    return {
        "overlap_fraction": count / total,
        "overlap_count": count,
        "mirror_ngram_total": total,
    }


def mirror_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.match(r"^mirror_\d+$", c)]


def analyze_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    mirrors = mirror_columns(df)
    if not mirrors:
        print(f"  [skip] no mirror_\\d+ columns in {path}")
        return pd.DataFrame()

    records = []
    for row_idx, row in df.iterrows():
        human_tokens = tokenize(parse_text(row["human"]))
        for mcol in mirrors:
            mirror_tokens = tokenize(parse_text(row[mcol]))
            for n in N_RANGE:
                stats = overlap_fraction(human_tokens, mirror_tokens, n)
                records.append(
                    {
                        "csv_file": path,
                        "mirror_col": mcol,
                        "row_idx": row_idx,
                        "n": n,
                        **stats,
                    }
                )
    return pd.DataFrame(records)


def plot_file(sub: pd.DataFrame, csv_path: str, mirror_col: str, out_dir: str):
    """Boxplot of overlap_fraction distribution across rows, one box per n."""
    fig, ax = plt.subplots(figsize=(12, 5))
    data_by_n = [sub[sub["n"] == n]["overlap_fraction"].dropna().values for n in N_RANGE]
    ax.boxplot(data_by_n, positions=N_RANGE, widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="#a8d8ea", alpha=0.8),
               medianprops=dict(color="navy", linewidth=2))
    ax.set_xlabel("N (n-gram size)", fontsize=12)
    ax.set_ylabel("Fraction of mirror n-grams\nfound verbatim in human", fontsize=11)
    ax.set_title(f"{os.path.basename(csv_path)}  ·  {mirror_col}", fontsize=13)
    ax.set_xticks(N_RANGE)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    parent = os.path.basename(os.path.dirname(csv_path))
    fname = f"{parent}__{stem}__{mirror_col}_overlap.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_aggregate(summary: pd.DataFrame, out_dir: str):
    """Median overlap_fraction vs n, one line per mirror_col, aggregated over all CSVs."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for mcol, grp in summary.groupby("mirror_col"):
        med = grp.groupby("n")["overlap_fraction"].median()
        ax.plot(med.index, med.values, marker="o", label=mcol)
    ax.set_xlabel("N (n-gram size)", fontsize=12)
    ax.set_ylabel("Median overlap fraction (all CSVs)", fontsize=11)
    ax.set_title("Aggregate: verbatim n-gram overlap — all CSVs", fontsize=13)
    ax.set_xticks(N_RANGE)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="mirror col")
    ax.grid(alpha=0.3)
    fname = os.path.join(out_dir, "aggregate_trend.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved aggregate plot → {fname}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    all_csvs = []
    for pattern in CSV_GLOBS:
        all_csvs.extend(sorted(glob.glob(pattern)))

    if not all_csvs:
        print("No CSV files found. Run from the inference_set_rewrite directory.")
        sys.exit(1)

    print(f"Found {len(all_csvs)} CSV files.")
    all_records = []

    for path in all_csvs:
        print(f"\nProcessing: {path}")
        result = analyze_csv(path)
        if result.empty:
            continue
        all_records.append(result)

        # Per-file plots
        for mcol in result["mirror_col"].unique():
            sub = result[result["mirror_col"] == mcol]
            fname = plot_file(sub, path, mcol, PLOT_DIR)
            n_rows = sub["row_idx"].nunique()
            med1 = sub[sub["n"] == 1]["overlap_fraction"].median()
            med10 = sub[sub["n"] == 10]["overlap_fraction"].median()
            print(f"  {mcol}: {n_rows} rows | median overlap n=1: {med1:.3f}, n=10: {med10:.3f} → {fname}")

    if not all_records:
        print("No data collected.")
        return

    summary = pd.concat(all_records, ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary → {SUMMARY_CSV}  ({len(summary):,} rows)")

    plot_aggregate(summary, PLOT_DIR)

    # Print high-overlap rows (n=10, overlap > 0.5) as a quick sanity check
    flagged = summary[(summary["n"] == 10) & (summary["overlap_fraction"] > 0.5)]
    if not flagged.empty:
        print(f"\n⚠ {len(flagged)} row×mirror pairs with >50% overlap at n=10:")
        print(flagged[["csv_file", "mirror_col", "row_idx", "overlap_fraction"]].to_string(index=False))
    else:
        print("\nNo rows with >50% overlap at n=10 (good).")


if __name__ == "__main__":
    main()
