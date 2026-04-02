"""
Heatmap: mean Pangram/BERT score as a function of
  x-axis — n  (n-gram size)
  y-axis — threshold  (max allowed count of overlapping n-grams between mirror and human)

For each cell (n, threshold): keep only rows where overlap_count at that n <= threshold,
then compute mean score over those rows.

Requires ngram_overlap_summary.csv (from analyze_ngram_overlap.py) OR recomputes on the fly.
"""

import ast
import re
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Config ─────────────────────────────────────────────────────────────────────
N_RANGE = list(range(1, 25))      # x-axis: n-gram sizes
MAX_THRESHOLD = 51                 # y-axis: 0 .. MAX_THRESHOLD (inclusive)
CSV_GLOBS = [
    "iterative_prompt_rewrite_test/results_*.csv",
    "iterative_prompt_rewrite_test_pangram/results_*.csv",
    # "old_csvs/results_*.csv",
]
OVERLAP_SUMMARY = "ngram_overlap_summary.csv"
PLOT_DIR = "threshold_sensitivity_plots"
OUT_SUMMARY = "threshold_sensitivity_summary.csv"
# ───────────────────────────────────────────────────────────────────────────────


# ── Text / overlap helpers ──────────────────────────────────────────────────────

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


def tokenize(text: str) -> list:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def mirror_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if re.match(r"^mirror_\d+$", c)]


def compute_overlap_counts(path: str) -> pd.DataFrame:
    """Compute per-row overlap_count for every n in N_RANGE and all mirror cols."""
    df = pd.read_csv(path)
    mirrors = mirror_columns(df)
    records = []
    for row_idx, row in df.iterrows():
        h_tok = tokenize(parse_text(row["human"]))
        for mcol in mirrors:
            m_tok = tokenize(parse_text(row[mcol]))
            for n in N_RANGE:
                if len(h_tok) < n or len(m_tok) < n:
                    count = 0
                else:
                    h_set = set(tuple(h_tok[i:i+n]) for i in range(len(h_tok)-n+1))
                    m_grams = [tuple(m_tok[i:i+n]) for i in range(len(m_tok)-n+1)]
                    count = sum(1 for g in m_grams if g in h_set)
                records.append({
                    "csv_file": path, "mirror_col": mcol,
                    "row_idx": row_idx, "n": n, "overlap_count": count,
                })
    return pd.DataFrame(records)


# ── Score column detection ──────────────────────────────────────────────────────

def score_col_for(df: pd.DataFrame, mirror_col: str):
    if "fraction_ai" in df.columns:
        df["ai_ai_assisted"] = df['fraction_ai'] + .5 * df['fraction_ai_assisted']
        return "ai_ai_assisted"
    t = re.search(r"(\d+)$", mirror_col)
    if t:
        candidate = f"mirror_{t.group(1)}_score_avg"
        if candidate in df.columns:
            return candidate
    return None


# ── Heatmap builder ─────────────────────────────────────────────────────────────

def build_heatmap(ov: pd.DataFrame, score_df: pd.DataFrame, mirror_col: str):
    """
    Returns matrix of shape (MAX_THRESHOLD+1, len(N_RANGE)):
      matrix[thresh, n_idx] = mean score for rows where overlap_count at n <= thresh

    Also returns n_kept matrix of same shape.
    """
    scol = score_col_for(score_df, mirror_col)
    if scol is None:
        return None, None, None

    scores_by_row = score_df[scol].reset_index().rename(columns={"index": "row_idx"})

    thresholds = np.arange(0, MAX_THRESHOLD + 1)   # 0..51
    mat_score = np.full((len(thresholds), len(N_RANGE)), np.nan)
    mat_kept  = np.zeros((len(thresholds), len(N_RANGE)), dtype=int)

    for n_idx, n in enumerate(N_RANGE):
        ov_n = ov[ov["n"] == n][["row_idx", "overlap_count"]].merge(
            scores_by_row, on="row_idx", how="inner"
        ).dropna(subset=[scol])

        for t_idx, thresh in enumerate(thresholds):
            kept = ov_n[ov_n["overlap_count"] <= thresh]
            mat_kept[t_idx, n_idx] = len(kept)
            if len(kept) > 0:
                mat_score[t_idx, n_idx] = kept[scol].mean()

    return mat_score, mat_kept, scol


# ── Plot ────────────────────────────────────────────────────────────────────────

def plot_heatmap(mat_score, mat_kept, csv_path, mirror_col, scol):
    thresholds = np.arange(0, MAX_THRESHOLD + 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             gridspec_kw={"width_ratios": [3, 1]})

    # ── Left: mean score heatmap ──
    ax = axes[0]
    # Mask cells with no data
    masked = np.ma.masked_where(np.isnan(mat_score), mat_score)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#dddddd")

    im = ax.imshow(masked, aspect="auto", origin="lower",
                   cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, label=f"Mean {scol}")

    ax.set_xlabel("n  (n-gram size)", fontsize=12)
    ax.set_ylabel("Threshold  (max overlapping n-grams allowed)", fontsize=12)
    ax.set_xticks(range(len(N_RANGE)))
    ax.set_xticklabels(N_RANGE, fontsize=8)
    # Show every 5th threshold tick to avoid crowding
    tick_positions = list(range(0, len(thresholds), 5))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(thresholds[tick_positions], fontsize=8)
    title = f"{os.path.basename(csv_path)}  ·  {mirror_col}"
    ax.set_title(title, fontsize=13)

    # ── Right: rows kept heatmap ──
    ax2 = axes[1]
    masked_kept = np.ma.masked_where(mat_kept == 0, mat_kept)
    cmap2 = plt.get_cmap("Blues").copy()
    cmap2.set_bad(color="#eeeeee")
    im2 = ax2.imshow(masked_kept, aspect="auto", origin="lower",
                     cmap=cmap2, interpolation="nearest")
    plt.colorbar(im2, ax=ax2, label="Rows kept")
    ax2.set_xlabel("n", fontsize=12)
    ax2.set_xticks(range(len(N_RANGE)))
    ax2.set_xticklabels(N_RANGE, fontsize=8)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(thresholds[tick_positions], fontsize=8)
    ax2.set_title("Rows kept", fontsize=12)

    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    parent = os.path.basename(os.path.dirname(csv_path))
    fname = os.path.join(PLOT_DIR, f"{parent}__{stem}__{mirror_col}__heatmap.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_aggregate_heatmap(all_matrices: list):
    """Average normalized score matrices across all (csv, mirror_col) pairs."""
    normed = []
    for mat, _ in all_matrices:
        lo, hi = np.nanmin(mat), np.nanmax(mat)
        if hi > lo:
            normed.append((mat - lo) / (hi - lo))

    if not normed:
        return
    agg = np.nanmean(np.stack(normed, axis=0), axis=0)

    thresholds = np.arange(0, MAX_THRESHOLD + 1)
    fig, ax = plt.subplots(figsize=(14, 8))
    masked = np.ma.masked_where(np.isnan(agg), agg)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, aspect="auto", origin="lower",
                   cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Normalized mean score (avg over all CSVs)")
    ax.set_xlabel("n  (n-gram size)", fontsize=12)
    ax.set_ylabel("Threshold  (max overlapping n-grams allowed)", fontsize=12)
    ax.set_xticks(range(len(N_RANGE)))
    ax.set_xticklabels(N_RANGE, fontsize=9)
    tick_positions = list(range(0, len(thresholds), 5))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(thresholds[tick_positions], fontsize=9)
    ax.set_title("Aggregate heatmap: normalized mean score vs. (n, threshold)", fontsize=13)
    fig.tight_layout()
    fname = os.path.join(PLOT_DIR, "aggregate_heatmap.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved aggregate heatmap → {fname}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    all_csvs = []
    for pattern in CSV_GLOBS:
        all_csvs.extend(sorted(glob.glob(pattern)))

    if not all_csvs:
        print("No CSV files found. Run from the inference_set_rewrite directory.")
        return
    print(f"Found {len(all_csvs)} CSV files.")

    # Load or compute overlap counts for all n in N_RANGE
    if os.path.exists(OVERLAP_SUMMARY):
        print(f"Loading {OVERLAP_SUMMARY} ...")
        ov_full = pd.read_csv(OVERLAP_SUMMARY)
        # Filter to only n values we need, and only the columns we use
        ov_full = ov_full[ov_full["n"].isin(N_RANGE)][
            ["csv_file", "mirror_col", "row_idx", "n", "overlap_count"]
        ].copy()
        # Check if overlap_count is available; fall back to recompute if not
        if "overlap_count" not in ov_full.columns or ov_full["overlap_count"].isna().all():
            print("  overlap_count missing in summary — recomputing ...")
            ov_full = None
    else:
        ov_full = None

    if ov_full is None:
        print(f"Computing overlap counts for n={N_RANGE[0]}..{N_RANGE[-1]} ...")
        parts = []
        for path in all_csvs:
            print(f"  {path}")
            parts.append(compute_overlap_counts(path))
        ov_full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if ov_full.empty:
        print("No overlap data. Exiting.")
        return

    summary_records = []
    all_matrices = []

    for path in all_csvs:
        print(f"\nProcessing: {path}")
        try:
            score_df = pd.read_csv(path)
        except Exception as e:
            print(f"  [skip] {e}")
            continue

        mirrors = mirror_columns(score_df)
        if not mirrors:
            print("  [skip] no mirror_\\d+ columns")
            continue

        for mcol in mirrors:
            ov_sub = ov_full[
                (ov_full["csv_file"] == path) & (ov_full["mirror_col"] == mcol)
            ]
            if ov_sub.empty:
                print(f"  [skip] no overlap data for {mcol}")
                continue

            mat_score, mat_kept, scol = build_heatmap(ov_sub, score_df, mcol)
            if mat_score is None:
                print(f"  [skip] no score column for {mcol}")
                continue

            all_matrices.append((mat_score, mat_kept))
            fname = plot_heatmap(mat_score, mat_kept, path, mcol, scol)
            print(f"  {mcol} ({scol}) → {fname}")

            # Collect summary rows
            thresholds = np.arange(0, MAX_THRESHOLD + 1)
            for t_idx, thresh in enumerate(thresholds):
                for n_idx, n in enumerate(N_RANGE):
                    summary_records.append({
                        "csv_file": path, "mirror_col": mcol,
                        "n": n, "threshold": int(thresh),
                        "mean_score": mat_score[t_idx, n_idx],
                        "n_kept": int(mat_kept[t_idx, n_idx]),
                    })

    if not summary_records:
        print("No data collected.")
        return

    pd.DataFrame(summary_records).to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved summary → {OUT_SUMMARY}")

    plot_aggregate_heatmap(all_matrices)


if __name__ == "__main__":
    main()
