"""
Reader for the 2025 back-half arXiv parquet, for PU / TEDn training.

Setup requested:
  - labeled positives (P)  = 2025 LLM *mirrors* (one rewrite per row)
  - unlabeled (U)          = 2025 'human_abstract' column (itself an unknown
                             human/AI mixture) + an injected `alpha` fraction of
                             extra mirrors.  alpha = 0.25 by default.

This is a NEW file -- it does not modify any existing reader.  It mirrors the
sentence-level style of `data_helper.IMDb.read_arxiv_split2`: it returns a flat
(texts, labels) pair where label 1 == labeled positive and label 0 == unlabeled,
ready to hand to `IMDbBERTData` + `get_PUDataSplits1`.

The 2025 back-half parquet has exactly one non-null mirror column per row
(2500 rows each of 4 LLMs), so we partition rows into disjoint pools:
  - `label_rows`  : their mirror becomes a labeled positive  (P)
  - `inject_rows` : their mirror becomes an unlabeled positive (U contamination)
  - ALL rows      : their human_abstract becomes an unlabeled negative (U)
This keeps the labeled-positive mirrors and the injected unlabeled-positive
mirrors disjoint, matching the spirit of read_arxiv_split2.
"""

import os
import re

import numpy as np
import pandas as pd

from data_helper.IMDb import split_into_sentences

LLM_COLS_2025 = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]

# Every abstract that appears anywhere in the downstream analysis
# (pu/high_conf_human_analysis_cs) is listed here, one normalized abstract per
# line, so we can guarantee none of it leaks into training.  Rebuild with
# scripts/train_2025/build_exclusion.py if the downstream set changes.
DEFAULT_EXCLUDE_PATH = os.path.join(os.path.dirname(__file__), "downstream_exclude_norm.txt")


def _norm(s):
    """Whitespace/punctuation-insensitive, lowercased key for matching abstracts."""
    return re.sub(r"[^a-z0-9]", "", s.lower()) if isinstance(s, str) else ""


def load_exclusion(path=DEFAULT_EXCLUDE_PATH):
    """Load the set of normalized downstream abstracts to exclude from training."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"downstream exclusion list not found at {path}; "
            f"run scripts/train_2025/build_exclusion.py first")
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def _row_mirror(row):
    """Return the single non-null LLM mirror for a row (one per row in this file)."""
    for c in LLM_COLS_2025:
        v = row[c]
        if isinstance(v, str) and len(v) > 0:
            return v
    return None


def read_arxiv_2025_backhalf(split_dir, alpha, split, sentence, seed,
                             inject_frac=0.4, n_train_abstracts=2500, n_val_abstracts=834,
                             exclude_path=DEFAULT_EXCLUDE_PATH):
    """
    Args:
        split_dir        : path to arxiv_2025_ai_cs._10000_backhalf.parquet
        alpha            : target fraction of the unlabeled pool that is injected mirrors.
                           alpha=0 means the unlabeled pool is purely 2025 'human' text
                           (no manually inserted AI rewrites) and every mirror is a
                           labeled positive.
        split            : "train" | "val"
        sentence         : True for sentence-level (matches run_arxiv.py), False for abstract-level
        seed             : controls the row shuffle / partition and the alpha subsampling
        inject_frac      : fraction of rows whose mirror is reserved for U contamination
                           (only used when alpha > 0)
        n_train_abstracts: # abstracts feeding the train split. Default 2500 matches the
                           per-split abstract count of run_arxiv.py's read_arxiv_split2
                           (num_inject=.2*10k -> wrong[:1250]+right[:1250]).
        n_val_abstracts  : # abstracts feeding the val split. Default 834 matches
                           read_arxiv_split2's val accounting (417 wrong + 417 right).
                           Train and val abstract pools are disjoint.
        exclude_path     : file of normalized downstream abstracts to drop (leakage guard)

    Returns:
        (texts, labels) with label 1 == labeled positive, 0 == unlabeled.
    """
    assert seed is not None
    assert split in ("train", "val")

    arxiv_data = pd.read_parquet(split_dir)
    arxiv_data = arxiv_data.assign(
        mirror=[_row_mirror(arxiv_data.iloc[i]) for i in range(len(arxiv_data))]
    )
    arxiv_data = arxiv_data[arxiv_data["mirror"].notnull()].reset_index(drop=True)

    # --- leakage guard: drop any abstract used anywhere in the downstream analysis ---
    exclude = load_exclusion(exclude_path)
    before = len(arxiv_data)
    keep_mask = [_norm(a) not in exclude for a in arxiv_data["human_abstract"]]
    arxiv_data = arxiv_data[keep_mask].reset_index(drop=True)
    print(f"[2025 {split}] leakage guard: dropped {before - len(arxiv_data)} / {before} "
          f"abstracts overlapping pu/high_conf_human_analysis_cs")

    # deterministic per-seed shuffle, then fixed-size disjoint train/val row pools
    # (sized to match run_arxiv.py's read_arxiv_split2 scale, not the full 10k)
    arxiv_data = arxiv_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    assert n_train_abstracts + n_val_abstracts <= len(arxiv_data), \
        f"need {n_train_abstracts + n_val_abstracts} abstracts, have {len(arxiv_data)}"
    if split == "train":
        subset = arxiv_data.iloc[:n_train_abstracts].reset_index(drop=True)
    else:  # val: the next disjoint block of rows
        subset = arxiv_data.iloc[n_train_abstracts:n_train_abstracts + n_val_abstracts].reset_index(drop=True)

    # disjoint partition of rows: injection pool vs labeled-positive pool.
    # alpha == 0 -> no injection: every mirror is a labeled positive.
    n_inject = 0 if alpha <= 0 else int(inject_frac * len(subset))
    inject_rows = subset.iloc[:n_inject].reset_index(drop=True)
    label_rows = subset.iloc[n_inject:].reset_index(drop=True)

    positive_texts = label_rows["mirror"].dropna().tolist()   # labeled P
    u_positive_pool = inject_rows["mirror"].dropna().tolist()  # injected unlabeled P
    u_negative_texts = subset["human_abstract"].dropna().tolist()  # unlabeled (all humans)

    if sentence:
        positive_texts, _ = split_into_sentences(positive_texts, [0] * len(positive_texts))
        u_positive_pool, _ = split_into_sentences(u_positive_pool, [0] * len(u_positive_pool))
        u_negative_texts, _ = split_into_sentences(u_negative_texts, [0] * len(u_negative_texts))

    # size the unlabeled pool so injected mirrors are exactly `alpha` of it
    rng = np.random.default_rng(seed)
    T_pos = len(u_positive_pool) / alpha if alpha > 0 else np.inf
    T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else np.inf
    T = int(min(T_pos, T_neg))
    n_pos = int(alpha * T)
    n_neg = T - n_pos

    u_positive_texts = list(rng.choice(u_positive_pool, size=n_pos, replace=False)) if n_pos > 0 else []
    u_negative_texts = list(rng.choice(u_negative_texts, size=n_neg, replace=False)) if n_neg > 0 else []

    realized = len(u_positive_texts) / max(1, len(u_positive_texts) + len(u_negative_texts))
    print(f"[2025 {split}] |P|={len(positive_texts)}  |U|={len(u_positive_texts) + len(u_negative_texts)} "
          f"(alpha={realized:.3f} target={alpha})")

    texts = positive_texts + u_positive_texts + u_negative_texts
    labels = [1] * len(positive_texts) + [0] * (len(u_positive_texts) + len(u_negative_texts))

    assert len(texts) == len(labels)
    return texts, labels
