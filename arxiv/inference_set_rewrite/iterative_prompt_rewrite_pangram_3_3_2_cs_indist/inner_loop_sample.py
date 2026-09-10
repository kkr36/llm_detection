"""Generate and Pangram-score a reproducible random 15-row validation sample."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import pandas as pd
from pangram import Pangram
from tqdm import tqdm

import strategy
from openai_api import openai_oss_query
from pangram_config import (
    PANGRAM_EXPECTED_VERSION,
    PANGRAM_EXPECTED_VERSION_PREFIX,
    PANGRAM_MODEL,
)
from util import collect_pangram_scores

SAMPLE_SIZE = 15

with open("/home/kkr36/creds.json", "r") as handle:
    pangram_api_key = json.load(handle)["pangram_api_key"]
pangram_client = Pangram(api_key=pangram_api_key)

query_fn = openai_oss_query
llm_label = "GPT OSS 120b"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default=strategy.CURRENT_STRATEGY.__name__,
        help="Name of a rewrite function defined in strategy.py.",
    )
    parser.add_argument(
        "--timestep",
        default=str(strategy.CURRENT_TIMESTEP),
        help="Timestep label used in the mirror column and output filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional sample seed; defaults to 5000 + 1000 * timestep.",
    )
    args = parser.parse_args()

    candidate = getattr(strategy, args.strategy, None)
    if not callable(candidate) or not args.strategy.startswith("rewrite_strategy_"):
        raise ValueError(f"unknown rewrite strategy: {args.strategy}")

    timestep = args.timestep
    sample_seed = (
        args.seed
        if args.seed is not None
        else 5000 + int(Decimal(timestep) * 1000)
    )
    output_csv = Path(f"results_{timestep}_oss_val_{SAMPLE_SIZE}.csv")
    if output_csv.exists():
        raise FileExistsError(f"refusing to overwrite {output_csv}")

    print(f"starting sampled validation generation for t={timestep}")
    print(f"strategy={args.strategy}")
    print(f"sample size={SAMPLE_SIZE}, seed={sample_seed}")
    print(f"scoring with Pangram model={PANGRAM_MODEL}")

    validation = pd.read_csv("results_0_oss_val_50.csv")
    assert len(validation) == 50
    sampled = validation.sample(n=SAMPLE_SIZE, random_state=sample_seed)
    validation_rows = sampled.index.tolist()
    source_abstracts = sampled["mirror_0"].tolist()
    print(f"validation rows={validation_rows}")

    generated = {}

    def process_row(position):
        source = source_abstracts[position]
        mirror, _ = candidate(query_fn, source, llm_label)
        return position, mirror, source

    print("generating mirrors")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_row, position): position
            for position in range(SAMPLE_SIZE)
        }
        for future in tqdm(as_completed(futures), total=SAMPLE_SIZE):
            position, mirror, source = future.result()
            generated[position] = (mirror, source)

    mirrors = [generated[position][0] for position in range(SAMPLE_SIZE)]
    sources = [generated[position][1] for position in range(SAMPLE_SIZE)]

    output = {
        "validation_row": validation_rows,
        "sample_seed": [sample_seed] * SAMPLE_SIZE,
        "rewrite_strategy": [args.strategy] * SAMPLE_SIZE,
        "original": sources,
        "mirroring_llm": [llm_label] * SAMPLE_SIZE,
        f"mirror_{timestep}": mirrors,
    }

    pangram_scores = collect_pangram_scores(
        pangram_client,
        mirrors,
        model=PANGRAM_MODEL,
        expected_version=PANGRAM_EXPECTED_VERSION,
        expected_version_prefix=PANGRAM_EXPECTED_VERSION_PREFIX,
    )
    output.update(pangram_scores)

    pd.DataFrame(output).to_csv(output_csv, index=False)
    print(f"saved {output_csv}")
