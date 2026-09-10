"""Experiment: does putting v16's sentences back into a LOGICAL order shift Pangram,
even though not a single sentence's wording is changed?

v16 deliberately scrambles discourse order (result-first, arc-broken). Here we:
1. split each of the 25 v16 rewrites into sentences,
2. ask GPT-OSS ONLY for a logical ordering (a permutation of sentence indices -- never
   any rewritten text),
3. reassemble the VERBATIM sentences in that order (hard-checked: the sentence multiset
   is identical, so no wording changes),
4. rescore the reordered abstract with Pangram 3.3.2 and compare to the v16 baseline.

Writes results_16_logical_reorder_pangram.csv. Does not touch any canonical result file.
Run in the llm_master env.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pangram import Pangram
from tqdm import tqdm

from openai_api import openai_oss_query
from pangram_config import (
    PANGRAM_EXPECTED_VERSION,
    PANGRAM_EXPECTED_VERSION_PREFIX,
    PANGRAM_MODEL,
)
from util import collect_pangram_scores, split_into_sentences

SRC = Path("results_16_oss_train_25_pangram.csv")
OUT = Path("results_16_logical_reorder_pangram.csv")
MAX_WORKERS = 10

_ORDER_CONTEXT = """
You are given numbered sentences from a scientific abstract. They are currently in a deliberately scrambled,
hard-to-follow order. Decide the order that reads most logically -- the natural flow of a scientific abstract:
background/motivation first, then the problem or gap, then the approach or method, then the results, then the
conclusion or implication.

Do NOT rewrite, merge, split, paraphrase, or edit any sentence in any way. Your only job is to choose an order.
Output ONLY the sentence numbers in your chosen order, separated by commas (for example: 4,1,5,2,3). Include every
number exactly once and output nothing else.
"""


def logical_permutation(sentences):
    """Return a 1-based permutation of sentence indices from the model, repaired to a
    valid full permutation. Falls back to identity for any missing indices."""
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    raw = openai_oss_query(_ORDER_CONTEXT, f"Sentences:\n{numbered}")
    n = len(sentences)
    seen, perm = set(), []
    for tok in re.findall(r"\d+", raw):
        i = int(tok)
        if 1 <= i <= n and i not in seen:
            seen.add(i)
            perm.append(i)
    for i in range(1, n + 1):  # append any indices the model dropped, in original order
        if i not in seen:
            perm.append(i)
    return perm, (len(seen) == n and len(perm) == n)


def reorder_verbatim(text):
    sents = [s.strip() for s in split_into_sentences(text) if s.strip()]
    if len(sents) <= 1:
        return text, list(range(1, len(sents) + 1)), True, len(sents)
    perm, model_valid = logical_permutation(sents)
    reordered_sents = [sents[i - 1] for i in perm]
    # HARD GUARANTEE: exact same sentences, only permuted -- no wording changed.
    assert sorted(reordered_sents) == sorted(sents), "sentence set changed!"
    return " ".join(reordered_sents), perm, model_valid, len(sents)


def main():
    df = pd.read_csv(SRC).reset_index(drop=True)
    n = len(df)
    results = {}

    def work(pos):
        text = str(df.loc[pos, "mirror_16"])
        reordered, perm, valid, nsent = reorder_verbatim(text)
        return pos, reordered, perm, valid, nsent

    print(f"reordering {n} v16 rewrites into logical order (verbatim sentences)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(work, p): p for p in range(n)}
        for f in tqdm(as_completed(futs), total=n):
            pos, reordered, perm, valid, nsent = f.result()
            results[pos] = (reordered, perm, valid, nsent)

    df["mirror_16_logical"] = [results[p][0] for p in range(n)]
    df["logical_perm"] = [json.dumps(results[p][1]) for p in range(n)]
    df["perm_model_valid"] = [results[p][2] for p in range(n)]
    df["n_sentences"] = [results[p][3] for p in range(n)]

    # final verbatim audit: a pure reorder must preserve the exact WORD multiset
    # (robust to spaCy re-segmenting the rejoined text on abbreviations/decimals).
    from collections import Counter

    changed = 0
    for p in range(n):
        a = Counter(str(df.loc[p, "mirror_16"]).split())
        b = Counter(str(df.loc[p, "mirror_16_logical"]).split())
        if a != b:
            changed += 1
            print(f"  WORD-LEVEL DIFF at row {p} (arxiv {df.loc[p,'arxiv_id']}): "
                  f"only-in-orig={list((a-b).elements())[:8]} only-in-new={list((b-a).elements())[:8]}")
    print(f"rows whose WORD multiset changed after reorder (must be 0): {changed}")
    assert changed == 0

    with open("/home/kkr36/creds.json") as fh:
        pangram = Pangram(api_key=json.load(fh)["pangram_api_key"])
    scores = collect_pangram_scores(
        pangram,
        df["mirror_16_logical"].tolist(),
        model=PANGRAM_MODEL,
        expected_version=PANGRAM_EXPECTED_VERSION,
        expected_version_prefix=PANGRAM_EXPECTED_VERSION_PREFIX,
        desc="scoring reordered",
    )
    for k, v in scores.items():
        df[f"logical_{k}"] = v

    def row_pai(ws):
        vals = [s for s in (ws or []) if isinstance(s, (int, float)) and not pd.isna(s)]
        return float(np.mean(vals)) if vals else np.nan

    df["pai_logical"] = df["logical_window_ai_assistance_scores"].map(row_pai)
    df.to_csv(OUT, index=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
