"""
add_num_xz_cols.py

Reads logging_accuracy_xz_frac.csv and adds two new columns:
  num_X  -- number of rewrite_X sentences in the training split
  num_Z  -- number of rewrite_Z sentences in the training split

Applies only to rows where _is_xz_float_col(f"rewrite_{train_llm}") is True,
which should be every row in this CSV.

Training data logic (mirrors helper.py / IMDb.py):
  PN  : pn_slice  = shuffled_data.iloc[4000:8000]  (4000 abstracts)
  PU  : neg_slice = shuffled_data.iloc[4000+int(4000*alpha):8000]
  frac = float parsed from train_llm, e.g. "xz_0.3" -> 0.3
  n_z  = int(round(n * frac))
  n_x  = n - n_z
  X abstracts: pn/neg_slice["rewrite_X"][:n_x]
  Z abstracts: pn/neg_slice["rewrite_Z"][:n_z]
  Sentence splitting is done via spacy (en_core_web_lg senter).

Usage:
  python add_num_xz_cols.py [--csv PATH] [--parquet PATH] [--seed INT]
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv",
    default=os.path.join(os.path.dirname(__file__), "logging_accuracy_xz_frac.csv"),
    help="Path to logging_accuracy_xz_frac.csv",
)
parser.add_argument(
    "--parquet",
    default="/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_xyz_cs._10000_fronthalf.parquet",
    help="Path to the arxiv 2020 xyz parquet used for xy training",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed used to shuffle the parquet (matches prepare_sliding_window.py)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load spacy (same setup as IMDb.py)
# ---------------------------------------------------------------------------
import en_core_web_lg

nlp = en_core_web_lg.load(disable=["ner", "parser"])
nlp.enable_pipe("senter")


def count_sentences_batch(texts, batch_size=200):
    """Count total spacy sentences across a list of texts."""
    total = 0
    for doc in nlp.pipe(texts, batch_size=batch_size):
        total += sum(1 for _ in doc.sents)
    return total


def per_abstract_sent_counts(texts, batch_size=200):
    """Return a list of sentence counts, one per abstract."""
    counts = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size),
        total=len(texts),
        desc="  spacy",
        leave=False,
    ):
        counts.append(sum(1 for _ in doc.sents))
    return counts


# ---------------------------------------------------------------------------
# Helper: replicate IMDb._is_xz_float_col and frac parsing
# ---------------------------------------------------------------------------
def _is_xz_float_col(llm_col):
    return bool(re.match(r"^rewrite_xz_(\d+(\.\d*)?|\.\d+)$", llm_col))


def _parse_frac(llm_col):
    m = re.search(r"(\d+(\.\d*)?|\.\d+)$", llm_col)
    if not m:
        raise ValueError(f"Cannot parse frac from: {llm_col}")
    return float(m.group(0))


# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
print(f"Loading CSV: {args.csv}")
df = pd.read_csv(args.csv)

# Sanity-check that every row has an xz_float train_llm
llm_cols = df["train_llm"].apply(lambda x: f"rewrite_{x}")
if not llm_cols.apply(_is_xz_float_col).all():
    bad = df.loc[~llm_cols.apply(_is_xz_float_col), "train_llm"].unique()
    print(f"WARNING: rows with non-xz_float train_llm (will be skipped): {bad}")

# ---------------------------------------------------------------------------
# Load and shuffle parquet (seed fixed to match training)
# ---------------------------------------------------------------------------
print(f"Loading parquet: {args.parquet}")
arxiv_data = pd.read_parquet(args.parquet)
arxiv_data = arxiv_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
print(f"  Total rows: {len(arxiv_data)}")

# ---------------------------------------------------------------------------
# Precompute per-abstract sentence counts for the two relevant slices:
#   PN  : iloc[4000:8000]  (always the same 4000 abstracts)
#   PU  : iloc[4000+int(4000*alpha):8000]  (varies by alpha)
#
# For each slice we need per-abstract counts for rewrite_X and rewrite_Z
# so that we can look up cumulative sums for any frac without re-running spacy.
# ---------------------------------------------------------------------------

# Identify unique (learning_method, train_alpha) combos that actually appear
unique_configs = (
    df[["learning_method", "train_alpha"]]
    .drop_duplicates()
    .to_dict("records")
)

# cache: (learning_method, train_alpha) -> {"x_cumsum": np.array, "z_cumsum": np.array, "n": int}
sent_cache = {}

for cfg in unique_configs:
    method = cfg["learning_method"]
    alpha  = float(cfg["train_alpha"])
    key    = (method, alpha)

    if method == "PN":
        data_slice = arxiv_data.iloc[4000:8000].reset_index(drop=True)
    else:
        # PU / TEDn
        neg_start  = 4000 + int(4000 * alpha)
        data_slice = arxiv_data.iloc[neg_start:8000].reset_index(drop=True)

    n = len(data_slice)
    print(f"\nPre-computing sentence counts for {method} alpha={alpha}  ({n} abstracts) ...")

    x_texts = data_slice["rewrite_X"].tolist()
    z_texts = data_slice["rewrite_Z"].tolist()

    print("  counting rewrite_X sentences ...")
    x_counts = per_abstract_sent_counts(x_texts)
    print("  counting rewrite_Z sentences ...")
    z_counts = per_abstract_sent_counts(z_texts)

    sent_cache[key] = {
        "n":        n,
        "x_cumsum": np.cumsum([0] + x_counts),  # length n+1; cumsum[k] = total sentences in first k abstracts
        "z_cumsum": np.cumsum([0] + z_counts),
    }

# ---------------------------------------------------------------------------
# For each row, look up num_X and num_Z
# ---------------------------------------------------------------------------

def get_num_xz(row):
    method    = row["learning_method"]
    alpha     = float(row["train_alpha"])
    llm_col   = f"rewrite_{row['train_llm']}"

    if not _is_xz_float_col(llm_col):
        return pd.Series({"num_X": np.nan, "num_Z": np.nan})

    frac = _parse_frac(llm_col)
    key  = (method, alpha)

    if key not in sent_cache:
        return pd.Series({"num_X": np.nan, "num_Z": np.nan})

    cache = sent_cache[key]
    n     = cache["n"]
    n_z   = int(round(n * frac))
    n_x   = n - n_z

    num_X = int(cache["x_cumsum"][n_x])
    num_Z = int(cache["z_cumsum"][n_z])

    return pd.Series({"num_X": num_X, "num_Z": num_Z})


print("\nMapping counts back to CSV rows ...")
counts = df.apply(get_num_xz, axis=1)
df["num_X"] = counts["num_X"].astype("Int64")
df["num_Z"] = counts["num_Z"].astype("Int64")

# ---------------------------------------------------------------------------
# Write updated CSV
# ---------------------------------------------------------------------------
df.to_csv(args.csv, index=False)
print(f"\nDone. Updated CSV written to: {args.csv}")
print(df[["learning_method", "train_llm", "train_alpha", "num_X", "num_Z"]].drop_duplicates())
