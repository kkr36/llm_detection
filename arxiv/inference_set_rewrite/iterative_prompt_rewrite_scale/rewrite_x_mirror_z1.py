"""
rewrite_x_mirror_z1.py
----------------------
Parallel driver (does NOT touch the xy / xyz / fd artifacts).

Applies the Z adversarial strategy at ITERATION 1 to the naive-AI column
`rewrite_X` (a "mirror" of rewrite_X), for a given method (PU or PN). This mirrors
the tuning setup in iterative_prompt_rewrite_test_fastdetect_gpt, where the
adversarial prompt was applied to AI text (mirror_0), not to human_abstract. The
human side of the eventual training pair stays `human_abstract`, which is carried
through unchanged.

Method is taken from argv so PU and PN can be launched as two independent jobs that
write to two separate files (no shared-write race):

    python rewrite_x_mirror_z1.py PU
    python rewrite_x_mirror_z1.py PN

Input : arxiv_{year}_xy_{category}_{half}_fronthalf.parquet   (read-only)
Output: arxiv_{year}_xmirror_z1_{method}_{category}_{half}_fronthalf.parquet   (new)
        adds column rewrite_Z_1_{method} = Z@iter1 applied to rewrite_X.
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

    iteration = 1                       # "adversarial prompt Z at timestep 1"
    method = sys.argv[1] if len(sys.argv) > 1 else "PU"
    assert method in ("PU", "PN"), f"method must be PU or PN, got {method!r}"
    strategy_fn = z_mapping[method][iteration]
    full_col_name = f"Z_{iteration}_{method}"      # -> column rewrite_Z_1_PU / rewrite_Z_1_PN

    source_col = "rewrite_X"            # mirror the naive-AI column, NOT human_abstract

    for year in tqdm(train_years, desc="years"):
        # Read the xy base (read-only). It carries human_abstract, rewrite_X, rewrite_Y.
        in_path = (
            f"/share/garg/arxiv_kaggle/multillm/data_raw/"
            f"arxiv_{year}_xy_{category}_{subsample_size//2}_fronthalf.parquet"
        )
        arxiv_data = pd.read_parquet(in_path)

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
                desc=f"mirror rewrite_X -> rewrite_{full_col_name} ({method})",
            ):
                i, new_text = fut.result()
                rewrites[i] = new_text

        arxiv_data[f"rewrite_{full_col_name}"] = rewrites

        # Parallel, per-method output file. Never overwrites xy / xyz / fd.
        out_path = (
            f"/share/garg/arxiv_kaggle/multillm/data_raw/"
            f"arxiv_{year}_xmirror_z{iteration}_{method}_{category}_{subsample_size//2}_fronthalf.parquet"
        )
        arxiv_data.to_parquet(out_path)
        print(f"saved {out_path}")
