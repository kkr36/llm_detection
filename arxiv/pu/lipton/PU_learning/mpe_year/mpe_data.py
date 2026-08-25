"""
Shared data plumbing for the mpe_year experiment.

Goal: estimate the proportion of AI-written text in *real* arXiv abstracts for
years 2020, 2023, 2025, with three methods (Pangram, PU/TEDn, James's method),
all trained and evaluated on the SAME unit of text: fixed 75-WORD CHUNKS.

Why chunks (not sentences / whole abstracts): to make the three methods directly
comparable, every method must score the *identical* pieces of text. Pangram needs
>= 50 words per call, PU/James used to work on sentences -- incompatible units.
So we chunk every abstract (human and AI) into 75-word segments and run all three
methods on exactly those chunks.

Chunking (CHUNK_SIZE=75, applied identically everywhere):
    - whitespace-split the text into words
    - group into consecutive 75-word chunks
    - the trailing remainder is MERGED INTO THE LAST chunk (last chunk = 75..149
      words), so no text is discarded; an abstract with < 75 words is one chunk.

Data (per year):
    /share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_cs._10000_fronthalf.parquet
Each of the 10,000 rows has:
    - human_abstract                      : the real (human) abstract
    - exactly one of LLM_COLS populated   : an LLM *mirror* rewrite of that abstract

Split convention:
    - TRAIN : first  N_TRAIN (8,000) rows  (chunks of human_abstract + AI mirror)
    - EVAL  : last   N_EVAL  (2,000) rows   (chunks of human_abstract only)

Only reads data + defines the chunker; modifies no existing file. Pure
pandas/numpy (no torch / spaCy), so it imports cleanly in every method's env.
"""

import re

import numpy as np
import pandas as pd

DATA_DIR = "/share/garg/arxiv_kaggle"
YEARS = [2020, 2023, 2025]

# The 4 LLM mirror columns present in every arxiv_{year}_ai_cs._10000_fronthalf.parquet
# (verified identical across 2020/2023/2025).
LLM_COLS = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]

N_TRAIN = 8000     # first 8,000 rows -> training pool
N_EVAL = 2000      # last  2,000 rows -> held-out evaluation (human_abstract only)
CHUNK_SIZE = 75    # words per chunk
EVAL_N_CHUNKS = 1000   # random subsample of test chunks scored per year (shared by all methods)
EVAL_SEED = 42         # fixed so every method scores the identical 1,000 chunks


def year_parquet(year):
    return f"{DATA_DIR}/multillm/data_raw/arxiv_{year}_ai_cs._10000_fronthalf.parquet"


# --------------------------------------------------------------------------- #
# 75-word chunking -- the single shared unit of analysis for all three methods.
# --------------------------------------------------------------------------- #
def chunk_75(text, size=CHUNK_SIZE):
    """Whitespace-split `text` into consecutive `size`-word chunks, merging the
    trailing remainder into the last chunk (so the last chunk is size..2*size-1
    words; a text with < size words is a single chunk). Returns chunk strings."""
    if not isinstance(text, str):
        return []
    words = text.split()
    if not words:
        return []
    n_full = len(words) // size
    if n_full <= 1:
        return [" ".join(words)]  # < 2*size words -> one chunk (whole text)
    chunks = []
    for i in range(n_full):
        start = i * size
        end = (i + 1) * size if i < n_full - 1 else len(words)  # last absorbs remainder
        chunks.append(" ".join(words[start:end]))
    return chunks


def chunks_of(abstracts, size=CHUNK_SIZE):
    """Flatten a list of abstracts into one list of 75-word chunk strings."""
    out = []
    for a in abstracts:
        out.extend(chunk_75(a, size))
    return out


def _word_tokenize(text):
    """Non-numeric lowercased word tokens (matches model_inference.tokenize_fn word split)."""
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if not w.isdigit()]


def chunk_tokens_of(abstracts, size=CHUNK_SIZE):
    """Same 75-word chunks as chunks_of(), each tokenized to a word list (for James)."""
    return [_word_tokenize(c) for c in chunks_of(abstracts, size)]


def tokens_of(chunks):
    """Tokenize already-formed chunk strings to word lists (for James), so James can
    score the exact same chunk strings the other methods do."""
    return [_word_tokenize(c) for c in chunks]


# --------------------------------------------------------------------------- #
# Row loading / splits
# --------------------------------------------------------------------------- #
def _row_mirror(row):
    """The single non-null LLM mirror for a row (one model per row in this file)."""
    for c in LLM_COLS:
        v = row[c]
        if isinstance(v, str) and len(v) > 0:
            return v
    return None


def load_split(year):
    """(train_df, eval_df): first N_TRAIN rows and last N_EVAL rows, no shuffle."""
    df = pd.read_parquet(year_parquet(year))
    assert len(df) >= N_TRAIN + N_EVAL, f"{year}: need {N_TRAIN + N_EVAL} rows, have {len(df)}"
    train_df = df.iloc[:N_TRAIN].reset_index(drop=True)
    eval_df = df.iloc[len(df) - N_EVAL:].reset_index(drop=True)
    return train_df, eval_df


def train_human_ai_abstracts(train_df):
    """(human_abstracts, ai_mirror_abstracts) from the training rows."""
    human = train_df["human_abstract"].dropna().tolist()
    mirrors = [_row_mirror(train_df.iloc[i]) for i in range(len(train_df))]
    ai = [m for m in mirrors if isinstance(m, str) and len(m) > 0]
    return human, ai


def eval_human_abstracts(eval_df):
    return eval_df["human_abstract"].dropna().tolist()


def eval_ai_mirror_chunks(year, n=None, seed=EVAL_SEED):
    """75-word chunks of the *AI mirror* rewrites on the last-2,000 (eval) rows.
    Same full-rewrite distribution as the PU P-set, but held out from training --
    used to inject known-alpha AI contamination into the test U for BBE sanity checks."""
    _, eval_df = load_split(year)
    mirrors = [_row_mirror(eval_df.iloc[i]) for i in range(len(eval_df))]
    mirrors = [m for m in mirrors if isinstance(m, str) and len(m) > 0]
    chunks = chunks_of(mirrors)
    if n is not None and len(chunks) > n:
        idx = np.random.default_rng(seed).choice(len(chunks), size=n, replace=False)
        chunks = [chunks[i] for i in sorted(idx)]
    return chunks


def eval_human_chunks(year, n=EVAL_N_CHUNKS, seed=EVAL_SEED):
    """The shared held-out test set: a fixed random sample of `n` 75-word chunks of
    the last-2,000 human abstracts. The sample is deterministic (fixed `seed`) so
    every method (Pangram / PU / James) is scored on exactly the same chunk strings.
    If fewer than `n` chunks exist, all are returned."""
    _, eval_df = load_split(year)
    chunks = chunks_of(eval_human_abstracts(eval_df))
    if n is not None and len(chunks) > n:
        idx = np.random.default_rng(seed).choice(len(chunks), size=n, replace=False)
        chunks = [chunks[i] for i in sorted(idx)]
    return chunks


# --------------------------------------------------------------------------- #
# PU / TEDn reader (chunk-level).
#   label 1 == labeled positive (AI mirror chunk)
#   label 0 == unlabeled        (human_abstract chunk, + optional alpha-fraction AI chunks)
# --------------------------------------------------------------------------- #
def read_fronthalf_pu(year, alpha, split, seed, val_frac=0.25, inject_frac=0.4):
    """
    (texts, labels) of 75-word chunks for PU/TEDn training, from the first N_TRAIN rows.

    alpha       : fraction of the unlabeled pool that is injected AI-mirror chunks.
                  alpha=0 -> unlabeled pool is purely human_abstract chunks (cleanest PU).
    split       : "train" | "val"  (disjoint row blocks within the first N_TRAIN rows)
    seed        : per-seed shuffle + subsampling
    val_frac    : fraction of the N_TRAIN rows held out for PU validation
    inject_frac : fraction of a split's rows whose mirror seeds the U pool (alpha>0 only)
    """
    assert split in ("train", "val")
    train_df, _ = load_split(year)

    df = train_df.assign(mirror=[_row_mirror(train_df.iloc[i]) for i in range(len(train_df))])
    df = df[df["mirror"].notnull()].reset_index(drop=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_val = int(len(df) * val_frac)
    subset = df.iloc[n_val:].reset_index(drop=True) if split == "train" \
        else df.iloc[:n_val].reset_index(drop=True)

    # disjoint partition: injection pool (mirror -> unlabeled positive) vs labeled-positive pool
    n_inject = 0 if alpha <= 0 else int(inject_frac * len(subset))
    inject_rows = subset.iloc[:n_inject].reset_index(drop=True)
    label_rows = subset.iloc[n_inject:].reset_index(drop=True)

    positive_texts = chunks_of(label_rows["mirror"].dropna().tolist())         # labeled P (AI)
    u_positive_pool = chunks_of(inject_rows["mirror"].dropna().tolist())        # injected unlabeled P (AI)
    u_negative_texts = chunks_of(subset["human_abstract"].dropna().tolist())   # unlabeled (human)

    # size U so injected mirror chunks are exactly `alpha` of it
    rng = np.random.default_rng(seed)
    T_pos = len(u_positive_pool) / alpha if alpha > 0 else np.inf
    T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else np.inf
    T = int(min(T_pos, T_neg))
    n_pos = int(alpha * T)
    n_neg = T - n_pos
    u_positive_texts = list(rng.choice(u_positive_pool, size=n_pos, replace=False)) if n_pos > 0 else []
    u_negative_texts = list(rng.choice(u_negative_texts, size=n_neg, replace=False)) if n_neg > 0 else []

    realized = len(u_positive_texts) / max(1, len(u_positive_texts) + len(u_negative_texts))
    print(f"[{year} {split}] |P|={len(positive_texts)}  |U|={len(u_positive_texts) + len(u_negative_texts)} "
          f"(alpha={realized:.3f} target={alpha})  [75-word chunks]")

    texts = positive_texts + u_positive_texts + u_negative_texts
    labels = [1] * len(positive_texts) + [0] * (len(u_positive_texts) + len(u_negative_texts))
    assert len(texts) == len(labels)
    return texts, labels
