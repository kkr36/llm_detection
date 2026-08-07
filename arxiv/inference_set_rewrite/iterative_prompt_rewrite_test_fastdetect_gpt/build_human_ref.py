# must be run using conda env *llm_embeddings* on a GPU node!
#
# One-off: build the fixed HUMAN reference pool that AUC compares the rewritten AI
# abstracts against. Fast-DetectGPT only yields a scalar d(x) per text, so measuring
# attack success needs a human d-distribution to compute AUC (separability of AI vs.
# human). We sample human_abstract rows from the SAME arXiv cs. parquet the AI seeds
# were generated from (same year / category / fronthalf, and restricted to rows where
# the seed LLM was generated, so the paper population matches), clean them the same way
# as the mirrors, score their d once, and cache to human_ref.csv.
#
# human_abstract is a different column from the LLM text, so there is no text overlap
# with the AI seeds by construction. Re-run only if you change the pool config below.
#
# Usage:  python build_human_ref.py            # ~300 humans, seed 0
#         python build_human_ref.py --n 500 --seed 1

import argparse
import sys

import numpy as np
import pandas as pd

from add_fastdetect import score_abstracts, sigmoid

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_inference import _llm_cols_and_parquet

# ---- reference-pool config ----
TEST_YEAR = 2020            # arXiv year the AI seeds come from (cs.)
SEED_LLM = "GPT OSS 120b"   # take humans from rows where this LLM was generated
CODEX = True                 # selects the ..._120b_qwen_codex.parquet that has GPT OSS 120b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of human abstracts to sample")
    ap.add_argument("--seed", type=int, default=0, help="sampling RNG seed (fixed pool)")
    ap.add_argument("--out", default="human_ref.csv")
    args = ap.parse_args()

    _, parquet_path = _llm_cols_and_parquet(TEST_YEAR, gemini=False, codex=CODEX)
    print(f"reading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # restrict to the same population the AI seeds were drawn from, then sample.
    pool = df[df[SEED_LLM].notna() & (df[SEED_LLM] != "")]
    pool = pool[pool["human_abstract"].notna() & (pool["human_abstract"] != "")]
    n = min(args.n, len(pool))
    sampled = pool.sample(n=n, random_state=args.seed)
    humans = sampled["human_abstract"].tolist()
    print(f"sampled {len(humans)} human abstracts from {len(pool)} candidates")

    d = score_abstracts(humans)
    out = pd.DataFrame({"human_abstract": humans, "d": d, "score_avg": sigmoid(d)})
    out.to_csv(args.out, index=False)

    finite = d[np.isfinite(d)]
    print(f"wrote {args.out}  (mean d={np.nanmean(d):.3f}, {len(finite)}/{len(d)} finite)")


if __name__ == "__main__":
    main()
