"""Correctness checks for the Fast-DetectGPT baseline plumbing (no GPU needed).

1. The refactored text readers reproduce get_u_data_llm / get_p_data_llm exactly.
2. Target polarity: UnlabelData.true_targets == 1 - u_labels (the get_metrics contract).
3. The dumped JSONs (if present) match the live pipeline element-wise.

Run on a compute node (spacy + parquet exceed login-node memory):
    srun --account=garg --partition=default_partition --mem=32G --cpus-per-task=4 --time=00:20:00 \
        /home/kkr36/.conda/envs/llm_embeddings/bin/python verify_fastdetect.py
"""

import json
import os
import sys

import numpy as np

from model_inference import (
    get_u_data_llm, get_p_data_llm,
    read_arxiv_positive_llm_texts, read_arxiv_unlabeled_llm_texts,
)

TEXTS_BASE = "/share/garg/arxiv_kaggle/fastdetect_texts"


def check(llm="Qwen", seed=0, codex=False, sentence=True):
    loader_u, u_texts_live, u_labels_live = get_u_data_llm(
        "ArXiv_BERT", 0.5, 2020, llm, sentence, True, False, True, "in", seed, codex=codex)
    loader_p = get_p_data_llm(
        "ArXiv_BERT", 2020, sentence, True, llm, False, True, seed, codex=codex)

    p_texts = read_arxiv_positive_llm_texts(2020, llm, sentence, True, False, True, seed, codex=codex)
    u_texts, u_labels = read_arxiv_unlabeled_llm_texts(
        0.5, 2020, llm, sentence, True, False, True, "in", seed, codex=codex)

    results = {
        "U texts identical": u_texts == u_texts_live,
        "U labels identical": u_labels == u_labels_live,
        "P len == loader dataset len": len(p_texts) == len(loader_p.dataset),
        "true_targets == 1 - u_labels":
            bool(np.array_equal(loader_u.dataset.true_targets, 1 - np.array(u_labels_live))),
    }

    # if the dump exists, it must match too
    gran = "sentence" if sentence else "abstract"
    dump = os.path.join(TEXTS_BASE, gran, "eval", llm.replace(" ", "_"), f"seed_{seed}.json")
    if os.path.exists(dump):
        with open(dump) as f:
            blob = json.load(f)
        results["dump U texts match live"] = blob["u_texts"] == u_texts_live
        results["dump U labels match live"] = blob["u_labels"] == u_labels_live
        results["dump P texts match refactor"] = blob["p_texts"] == p_texts

    print(f"--- {llm} seed={seed} codex={codex} sentence={sentence} ---")
    print(f"    |U|={len(u_texts)} |P|={len(p_texts)} pos_frac={np.mean(u_labels):.4f}")
    for k, v in results.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    return all(results.values())


if __name__ == "__main__":
    ok = True
    ok &= check("Qwen", 0, codex=False, sentence=True)
    ok &= check("Codex", 0, codex=True, sentence=True)
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
