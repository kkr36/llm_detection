"""Dump the exact test-set texts that the heatmap evaluation feeds to a detector.

Fast-DetectGPT is not a torch classifier, so `get_preds_llm` (which returns softmax
probabilities) is useless for it -- it needs the raw strings. This script writes out the
*final* strings, i.e. post `split_into_sentences` and post `clean_text`, which is exactly
what the DistilBert detectors see in prepare_heatmap.py / prepare_heatmap_codex.py.

Stage 1 of 3:
    1. dump_fastdetect_texts.py   (this file; CPU, needs spacy)
    2. fastdetect.py              (GPU, scores texts -> sha1-keyed cache)
    3. prepare_heatmap_fastdetect.py  (CPU, assembles metrics CSV)

Layout under OUT_BASE/<granularity>/:
    eval/<llm_slug>/seed_<n>.json    {p_texts, u_texts, u_labels}
    calib/<llm_slug>/seed_<n>.json   {human_texts, llm_texts}

`eval` is the held-out 25% block that the heatmap evaluates on. `calib` is drawn from the
complementary 75% (the block the supervised detectors trained on) and is used only to fit
the frozen source-LLM Platt scaler; it is disjoint from every eval U-set by construction.

Parquet convention mirrors prepare_heatmap_codex.py exactly: the Codex column reads the
codex parquet (12500 rows), every other column reads the standard parquet (10000 rows), so
the original columns reproduce the test sets the existing CSV already used.

Usage (env: /home/kkr36/.conda/envs/llm_embeddings):
    python dump_fastdetect_texts.py --granularity sentence
    python dump_fastdetect_texts.py --granularity abstract --llms "Llama 3.3 70b Instruct" Codex
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from model_inference import (
    _llm_cols_and_parquet,
    _select_llm_subset,
    read_arxiv_positive_llm_texts,
    read_arxiv_unlabeled_llm_texts,
)
from data_helper import split_into_sentences
from helper import clean_text

OUT_BASE = "/share/garg/arxiv_kaggle/fastdetect_texts"

# switches mirroring prepare_heatmap_codex.py
DATA_TYPE = "ArXiv_BERT"
TEST_YEAR = 2020
TEST_ALPHA = 0.5
CLEAN = True
GEMINI = False
EVAL_FLIP = True   # human = positive
SEEDS = 5

ORIG_LLMS = ["Qwen", "Gemini 3 Preview", "GPT OSS 120b", "Llama 3.3 70b Instruct"]
CODEX = "Codex"
ALL_LLMS = ORIG_LLMS + [CODEX]

# cap on calibration rows per side; a 2-parameter logistic needs nothing like the full 75%
N_CALIB_ROWS = 400


def slug(name):
    return name.replace(" ", "_")


def dump_eval(test_llm, seed, sentence, out_dir):
    """The held-out P-set and U-set for one (test_llm, seed), exactly as the heatmap uses them."""
    use_codex = (test_llm == CODEX)  # matches prepare_heatmap_codex.py

    p_texts = read_arxiv_positive_llm_texts(
        TEST_YEAR, test_llm, sentence, CLEAN, GEMINI, EVAL_FLIP, seed, codex=use_codex,
    )
    u_texts, u_labels = read_arxiv_unlabeled_llm_texts(
        TEST_ALPHA, TEST_YEAR, test_llm, sentence, CLEAN, GEMINI, EVAL_FLIP, "in", seed,
        codex=use_codex,
    )

    path = os.path.join(out_dir, "eval", slug(test_llm), f"seed_{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"p_texts": p_texts, "u_texts": u_texts, "u_labels": u_labels}, f)
    print(f"  eval  {test_llm} seed {seed}: |P|={len(p_texts)} |U|={len(u_texts)} "
          f"pos_frac={np.mean(u_labels):.3f} -> {path}")


def dump_calib(test_llm, seed, sentence, out_dir):
    """Labeled human/LLM texts from the *training* 75% block -- Platt source, disjoint from eval."""
    use_codex = (test_llm == CODEX)
    llm_cols, path_pq = _llm_cols_and_parquet(TEST_YEAR, GEMINI, use_codex)
    assert test_llm in llm_cols, f"{test_llm} not valid"

    arxiv_data = pd.read_parquet(path_pq)

    # _select_llm_subset returns the held-out tail of this exact shuffle; reproduce the shuffle
    # and take the complementary head, so calib and eval are disjoint at the row level.
    subset = arxiv_data[arxiv_data[test_llm].notna() & (arxiv_data[test_llm] != "")].reset_index(drop=True)
    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    cut = int(len(subset) * .75)
    train_block, eval_block = subset.iloc[:cut], subset.iloc[cut:]

    # positional-index disjointness (robust to duplicate abstract strings), plus a spot check
    # that eval_block really is what the eval path selects
    assert set(train_block.index).isdisjoint(set(eval_block.index)), "calib/eval row overlap"
    _, eval_human = _select_llm_subset(arxiv_data, llm_cols, test_llm, seed)
    assert eval_block["human_abstract"].tolist() == eval_human, \
        "calib shuffle does not reproduce the eval selection"

    train_block = train_block.iloc[:N_CALIB_ROWS]
    human_texts = train_block["human_abstract"].tolist()
    llm_texts = train_block[test_llm].tolist()

    if sentence:
        human_texts, _ = split_into_sentences(human_texts, [1] * len(human_texts))
        llm_texts, _ = split_into_sentences(llm_texts, [0] * len(llm_texts))

    if CLEAN:
        human_texts = clean_text(human_texts)
        llm_texts = clean_text(llm_texts)

    path = os.path.join(out_dir, "calib", slug(test_llm), f"seed_{seed}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"human_texts": human_texts, "llm_texts": llm_texts}, f)
    print(f"  calib {test_llm} seed {seed}: |human|={len(human_texts)} |llm|={len(llm_texts)} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["sentence", "abstract"], default="sentence")
    ap.add_argument("--llms", nargs="*", default=ALL_LLMS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--skip-calib", action="store_true",
                    help="eval sets only (used for the abstract-level validation run)")
    args = ap.parse_args()

    sentence = (args.granularity == "sentence")
    out_dir = os.path.join(OUT_BASE, args.granularity)

    for test_llm in args.llms:
        print(f"[{test_llm}]")
        for seed in range(args.seeds):
            dump_eval(test_llm, seed, sentence, out_dir)
            if not args.skip_calib:
                dump_calib(test_llm, seed, sentence, out_dir)

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
