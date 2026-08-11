"""
rewrite_x_mirror_z1.py
----------------------
In-place, single-parquet driver (same construction as rewrite_z_n_method.py):
reads ONE target parquet, computes the Z adversarial mirror of the naive-AI column
`rewrite_X` for a given (method, iteration), ADDS it as a new column
`rewrite_Z_{iteration}_{method}`, and writes the parquet back to the SAME path.
Existing columns are preserved; re-running a given (method, iteration) just
overwrites that one column.

Reusable across iterations -- pass the iteration on the command line, so the same
script generates the v2 adversarial mirrors once those Z_2 prompts exist:

    python rewrite_x_mirror_z1.py PU        # method=PU, iteration=1 (default)
    python rewrite_x_mirror_z1.py PN 1      # method=PN, iteration=1
    python rewrite_x_mirror_z1.py PU 2      # method=PU, iteration=2 (v2 mirror)
    python rewrite_x_mirror_z1.py PN 2

NOTE: this is a read-modify-write on a single file, so run ONE (method, iteration)
at a time. Do NOT run two invocations against the same parquet concurrently, or the
later writer will clobber the other's freshly added column.

Mirrors `rewrite_X` (naive AI text), NOT human_abstract -- matching the tuning
setup where the adversarial prompt was applied to AI text. human_abstract is carried
through untouched as the human side of the eventual training pair.

Target (read + written in place):
    arxiv_{year}_xyz_v2_{category}_{half}_fronthalf.parquet
It must already exist and contain the `rewrite_X` column to mirror.
"""
import sys
import concurrent.futures
import pandas as pd
from tqdm import tqdm

from openai_api import openai_oss_query
from strategy import z_mapping

model_name = "GPT OSS 120b"

if __name__ == "__main__":
    subsample_size = 20000
    category = "cs."
    train_years = [2020]

    method = sys.argv[1] if len(sys.argv) > 1 else "PU"
    iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    assert method in ("PU", "PN"), f"method must be PU or PN, got {method!r}"
    assert iteration in z_mapping[method], (
        f"no Z strategy for method={method} iteration={iteration}; "
        f"available iterations: {sorted(z_mapping[method])}"
    )
    strategy_fn = z_mapping[method][iteration]
    new_col = f"rewrite_Z_{iteration}_{method}"   # e.g. rewrite_Z_1_PU, rewrite_Z_2_PN

    source_col = "rewrite_X"        # mirror the naive-AI column, NOT human_abstract

    for year in tqdm(train_years, desc="years"):
        # Single target parquet, read + written in place (same construction as
        # rewrite_z_n_method.py). Must already exist and contain `source_col`.
        target_path = (
            f"/share/garg/arxiv_kaggle/multillm/data_raw/"
            f"arxiv_{year}_xyz_v2_{category}_{subsample_size//2}_fronthalf.parquet"
        )
        arxiv_data = pd.read_parquet(target_path)
        assert source_col in arxiv_data.columns, (
            f"{target_path} has no '{source_col}' column to mirror"
        )

        n = len(arxiv_data)
        rewrites = [None] * n

        def process_row(i):
            src = arxiv_data[source_col].iloc[i]
            if not isinstance(src, str) or len(src.strip()) < 10:
                return i, None            # leave invalid/empty sources unrewritten
            new_text, _ = strategy_fn(openai_oss_query, src, model_name)
            return i, new_text

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(process_row, i): i for i in range(n)}
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=n,
                desc=f"mirror {source_col} -> {new_col}",
            ):
                i, new_text = fut.result()
                rewrites[i] = new_text

        # Add (or overwrite) just this one column, then write back to the SAME path.
        arxiv_data[new_col] = rewrites
        arxiv_data.to_parquet(target_path)
        print(f"saved column {new_col} into {target_path}")
