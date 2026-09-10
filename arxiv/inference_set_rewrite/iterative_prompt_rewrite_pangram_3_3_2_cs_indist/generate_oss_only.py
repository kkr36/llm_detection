"""Generate a shared validation sample with GPT-OSS 120b and perform no scoring."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import strategy
from openai_api import openai_oss_query

DEFAULT_SAMPLE_SIZE = 15
DEFAULT_SAMPLE_SEED = 91313
LLM_LABEL = "GPT OSS 120b"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    args = parser.parse_args()

    candidate = getattr(strategy, args.strategy, None)
    if not callable(candidate) or not args.strategy.startswith("rewrite_strategy_v"):
        raise ValueError(f"unknown versioned rewrite strategy: {args.strategy}")
    if args.sample_size < 1 or args.sample_size > 50:
        raise ValueError("sample size must be between 1 and 50")

    output_csv = Path(
        f"oss_only_{args.version}_val_{args.sample_size}.csv"
    )
    if output_csv.exists():
        raise FileExistsError(f"refusing to overwrite {output_csv}")

    validation = pd.read_csv("results_0_oss_val_50.csv")
    if len(validation) != 50:
        raise ValueError(f"expected 50 validation rows, found {len(validation)}")
    sampled = validation.sample(n=args.sample_size, random_state=args.seed)
    validation_rows = sampled.index.tolist()
    sources = sampled["mirror_0"].tolist()

    print(f"starting OSS-only generation for {args.version}")
    print(f"strategy={args.strategy}")
    print(f"sample size={args.sample_size}, seed={args.seed}")
    print(f"validation rows={validation_rows}")
    print("Pangram scoring is disabled in this runner")

    generated = {}

    def process_row(position):
        source = sources[position]
        mirror, _ = candidate(openai_oss_query, source, LLM_LABEL)
        return position, mirror

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_row, position): position
            for position in range(args.sample_size)
        }
        for future in tqdm(as_completed(futures), total=args.sample_size):
            position, mirror = future.result()
            generated[position] = mirror

    mirrors = [generated[position] for position in range(args.sample_size)]
    output = pd.DataFrame(
        {
            "validation_row": validation_rows,
            "sample_seed": [args.seed] * args.sample_size,
            "rewrite_strategy": [args.strategy] * args.sample_size,
            "source": sources,
            "mirroring_llm": [LLM_LABEL] * args.sample_size,
            f"mirror_{args.version}": mirrors,
        }
    )
    output.to_csv(output_csv, index=False)
    print(f"saved {output_csv}")
