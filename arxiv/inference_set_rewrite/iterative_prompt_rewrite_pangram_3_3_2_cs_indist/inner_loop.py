"""Generate and Pangram-score a timestep-specific 25-row train sample.

Each timestep evaluates the active strategy on a deterministic sample from the
same 75-row mirror_0 pool. The seed mapping preserves the completed t=1 run:
t=1 -> seed 2024, t=2 -> seed 2025, and so on.

Run this file using the ``llm_master`` conda environment.
"""

import argparse
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
from strategy import CURRENT_STRATEGY, CURRENT_TIMESTEP
from util import collect_pangram_scores

TRAIN_CSV = Path("results_0_oss_train_75.csv")
TRAIN_POOL_SIZE = 75
TRAIN_SAMPLE_SIZE = 25
TIMESTEP_SEED_OFFSET = 2023
LLM_LABEL = "GPT OSS 120b"
MAX_WORKERS = 10


def sample_seed_for_timestep(timestep):
    """Map integer timesteps to distinct deterministic sample seeds."""
    if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 1:
        raise ValueError(f"timestep must be an integer >= 1, got {timestep!r}")
    return TIMESTEP_SEED_OFFSET + timestep


def row_pai(window_scores):
    valid_scores = [
        score
        for score in (window_scores or [])
        if isinstance(score, (int, float)) and not pd.isna(score)
    ]
    return float(np.mean(valid_scores)) if valid_scores else np.nan


def resolve_strategy(timestep):
    """Select rewrite_function_{timestep} so strategy and seed always agree.

    Guarantees the invariant the protocol depends on: timestep t runs the prompt
    named rewrite_function_{t} on the deterministic sample seeded 2023 + t. This
    removes any reliance on the hand-set CURRENT_STRATEGY pointer when a --timestep
    is passed, so a batch of timesteps cannot silently run the wrong prompt.
    """
    expected_name = f"rewrite_function_{timestep}"
    strategy_fn = getattr(strategy_module, expected_name, None)
    if strategy_fn is None:
        raise ValueError(
            f"no {expected_name} defined in strategy.py for timestep {timestep}"
        )
    return strategy_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help=(
            "Timestep t to run. Selects rewrite_function_{t} and derives the "
            "sample seed as 2023 + t. Defaults to strategy.CURRENT_TIMESTEP."
        ),
    )
    args = parser.parse_args()

    timestep = args.timestep if args.timestep is not None else CURRENT_TIMESTEP
    sample_seed = sample_seed_for_timestep(timestep)
    strategy_fn = resolve_strategy(timestep)
    strategy_name = strategy_fn.__name__

    # Fail loud if the resolved prompt does not match the timestep it will label.
    if strategy_name != f"rewrite_function_{timestep}":
        raise ValueError(
            f"strategy/timestep mismatch: {strategy_name} vs timestep {timestep}"
        )

    mirror_column = f"mirror_{timestep}"
    output_csv = Path(
        f"results_{timestep}_oss_train_{TRAIN_SAMPLE_SIZE}_pangram.csv"
    )

    if output_csv.exists():
        raise FileExistsError(f"refusing to overwrite {output_csv}")

    train = pd.read_csv(TRAIN_CSV)
    if len(train) != TRAIN_POOL_SIZE:
        raise ValueError(
            f"expected {TRAIN_POOL_SIZE} rows in {TRAIN_CSV}, found {len(train)}"
        )
    if "mirror_0" not in train.columns:
        raise ValueError(f"missing mirror_0 column in {TRAIN_CSV}")

    sampled = train.sample(
        n=TRAIN_SAMPLE_SIZE,
        random_state=sample_seed,
    )
    train_rows = sampled.index.tolist()
    sampled = sampled.reset_index(drop=True)
    sources = sampled["mirror_0"].tolist()

    print(f"starting generation for t={timestep}")
    print(f"strategy={strategy_name}")
    print(f"train sample size={TRAIN_SAMPLE_SIZE}, seed={sample_seed}")
    print(f"train rows={train_rows}")
    print(f"scoring with Pangram model={PANGRAM_MODEL}")
    print("generating mirrors")

    generated = {}

    def process_row(position):
        mirror, _ = strategy_fn(
            openai_oss_query,
            sources[position],
            LLM_LABEL,
        )
        return position, mirror

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_row, position): position
            for position in range(TRAIN_SAMPLE_SIZE)
        }
        for future in tqdm(as_completed(futures), total=TRAIN_SAMPLE_SIZE):
            position, mirror = future.result()
            generated[position] = mirror

    mirrors = [generated[position] for position in range(TRAIN_SAMPLE_SIZE)]

    with open("/home/kkr36/creds.json", "r") as handle:
        pangram_api_key = json.load(handle)["pangram_api_key"]
    pangram_client = Pangram(api_key=pangram_api_key)

    pangram_scores = collect_pangram_scores(
        pangram_client,
        mirrors,
        model=PANGRAM_MODEL,
        expected_version=PANGRAM_EXPECTED_VERSION,
        expected_version_prefix=PANGRAM_EXPECTED_VERSION_PREFIX,
    )

    output = sampled.copy()
    output.insert(0, "train_row", train_rows)
    output.insert(1, "sample_seed", sample_seed)
    output.insert(2, "rewrite_strategy", strategy_name)
    output["mirroring_llm"] = LLM_LABEL
    output[mirror_column] = mirrors
    for key, values in pangram_scores.items():
        output[key] = values
    output[f"pai_t{timestep}"] = output["window_ai_assistance_scores"].map(row_pai)

    output.to_csv(output_csv, index=False)
    print(f"saved {output_csv}")
