### DO NOT OVERWRITE, EDIT, OR TOUCH ANYTHING IN THIS FILE ###
# must be run using conda env *llm_master*!  (Bedrock API only, no GPU)
#
# Phase 1 of the Fast-DetectGPT attack loop: REWRITE ONLY.
# Reads the AI seed abstracts from results_0_oss_{split}_{to_rewrite}.csv, applies the
# current rewrite strategy (rewrite_strategy_v{timestep} in strategy.py) via the GPT-OSS
# rewrite LLM, and writes results_{timestep}_oss_{split}_{to_rewrite}.csv with columns
#   original, mirroring_llm, mirror_{timestep}
# Scoring is a separate GPU phase -- see add_fastdetect.py.

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import strategy
from strategy import CURRENT_TIMESTEP as DEFAULT_TIMESTEP
from openai_api import openai_oss_query

query_fns = [openai_oss_query]
llm_labels = ["GPT OSS 120b"]
assert len(query_fns) == len(llm_labels)


def resolve_strategy(timestep):
    """Pick rewrite_strategy_v{timestep} from strategy.py; error if it's still the stub."""
    fn = getattr(strategy, f"rewrite_strategy_v{timestep}", None)
    if fn is None:
        raise ValueError(f"strategy.py has no rewrite_strategy_v{timestep}; define it first.")
    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", type=int, default=DEFAULT_TIMESTEP)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--to_rewrite", type=int, default=15)
    args = parser.parse_args()
    CURRENT_TIMESTEP = args.timestep
    split = args.split
    to_rewrite = args.to_rewrite
    CURRENT_STRATEGY = resolve_strategy(CURRENT_TIMESTEP)
    output_csv = f"results_{CURRENT_TIMESTEP}_oss_{split}_{to_rewrite}.csv"
    print(f"starting generation for t={CURRENT_TIMESTEP} ({split}, {to_rewrite})")

    assert to_rewrite == 15 if split == "val" else to_rewrite == 50

    fpath = "results_0_oss_val_15.csv" if split == "val" else "results_0_oss_test_50.csv"
    arxiv_data = pd.read_csv(fpath)
    orig_abstracts = arxiv_data["mirror_0"].tolist()
    llms = [i % len(llm_labels) for i in range(len(arxiv_data))]

    # generate to_rewrite mirrors
    print("generating mirrors")
    results = {}

    def process_row(i):
        orig_abs = orig_abstracts[i]
        new_abs, _ = CURRENT_STRATEGY(query_fns[llms[i]], orig_abs, llm_labels[llms[i]])
        return i, new_abs, orig_abs

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, i): i for i in range(to_rewrite)}
        for future in tqdm(as_completed(futures), total=to_rewrite):
            i, new_abs, orig_abs = future.result()
            results[i] = (new_abs, orig_abs)

    # preserve original row order
    new_text, orig_text = [], []
    for i in sorted(results):
        new_text.append(results[i][0])
        orig_text.append(results[i][1])

    # each row: the original AI abstract and its rewritten mirror at this timestep.
    # Fast-DetectGPT scoring happens in add_fastdetect.py (GPU); this file only rewrites.
    abstract_dict = {
        "original": [],
        "mirroring_llm": [],
        f"mirror_{CURRENT_TIMESTEP}": [],
    }
    for i, (orig_abstract, mirror_t) in enumerate(zip(orig_text, new_text)):
        abstract_dict["original"].append(orig_abstract)
        abstract_dict[f"mirror_{CURRENT_TIMESTEP}"].append(mirror_t)
        abstract_dict["mirroring_llm"].append(llm_labels[llms[i]])

    pd.DataFrame(abstract_dict).to_csv(output_csv, index=False)
    print(f"wrote {output_csv}")
