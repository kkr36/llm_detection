"""Evaluate v20 and v21 on the 25-abstract VAL set (results_0_oss_val_25.csv).

Both strategies run on the SAME 25 held-out abstracts (disjoint from the train pool),
so v20 (5-call) vs v21 (4-call compressed) is a clean apples-to-apples comparison with
no sampling-seed confound. This is an evaluation on a fixed set, not a canonical train
timestep, so it does not touch inner_loop's numbering.

Writes results_{20,21}_oss_val_25_pangram.csv. Run in the llm_master env.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pangram import Pangram
from tqdm import tqdm

import strategy as strategy_module
from openai_api import openai_oss_query
from pangram_config import (
    PANGRAM_EXPECTED_VERSION,
    PANGRAM_EXPECTED_VERSION_PREFIX,
    PANGRAM_MODEL,
)
from util import collect_pangram_scores

VAL_CSV = Path("results_0_oss_val_25.csv")
LLM_LABEL = "GPT OSS 120b"
MAX_WORKERS = 10


def row_pai(ws):
    vals = [s for s in (ws or []) if isinstance(s, (int, float)) and not pd.isna(s)]
    return float(np.mean(vals)) if vals else np.nan


def run_strategy(val, t, pangram):
    strat = getattr(strategy_module, f"rewrite_function_{t}")
    sources = val["mirror_0"].tolist()
    mirror_col = f"mirror_{t}"
    print(f"\n=== {strat.__name__} on VAL (n={len(sources)}) ===")

    def process(pos):
        mirror, _ = strat(openai_oss_query, sources[pos], LLM_LABEL)
        return pos, mirror

    generated = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process, p): p for p in range(len(sources))}
        for f in tqdm(as_completed(futs), total=len(sources), desc=f"gen v{t}"):
            pos, mirror = f.result()
            generated[pos] = mirror
    mirrors = [generated[p] for p in range(len(sources))]

    scores = collect_pangram_scores(
        pangram,
        mirrors,
        model=PANGRAM_MODEL,
        expected_version=PANGRAM_EXPECTED_VERSION,
        expected_version_prefix=PANGRAM_EXPECTED_VERSION_PREFIX,
        desc=f"score v{t}",
    )

    out = val.copy()
    out["rewrite_strategy"] = strat.__name__
    out["mirroring_llm"] = LLM_LABEL
    out[mirror_col] = mirrors
    for k, v in scores.items():
        out[k] = v
    out[f"pai_t{t}"] = out["window_ai_assistance_scores"].map(row_pai)
    dest = Path(f"results_{t}_oss_val_25_pangram.csv")
    out.to_csv(dest, index=False)
    print(f"saved {dest}")
    human = (out["pangram_prediction_short"] == "Human").sum()
    print(f"v{t}: Human {human}/{len(out)} ({human/len(out):.2f}), mean pai {out[f'pai_t{t}'].mean():.3f}")


def main():
    val = pd.read_csv(VAL_CSV).reset_index(drop=True)
    with open("/home/kkr36/creds.json") as fh:
        pangram = Pangram(api_key=json.load(fh)["pangram_api_key"])
    for t in (20, 21):
        run_strategy(val, t, pangram)


if __name__ == "__main__":
    main()
