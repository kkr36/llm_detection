import os
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import pandas as pd
from openai_api import openai_oss_query
from strategy import rewrite_strategy_FD

model_name = "GPT OSS 120b"

if __name__ == "__main__":
    subsample_size = 20000
    category = "cs."
    train_years = [2020]
    # FD = Fast-DetectGPT method (single-call "v1" humanizing rewrite); a different
    # method family than the PU/PN Z-strategies, so it gets its own column and its own
    # output parquet suffix ("_fd_") rather than writing into the xyz artifact.
    strategies = [("FD", rewrite_strategy_FD)]
    rewrite_col = "rewrite_FD"

    abstract_col = "rewrite_X"

    for year in tqdm(train_years):

        arxiv_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_xy_{category}_{subsample_size//2}_fronthalf.parquet"
        arxiv_data = pd.read_parquet(arxiv_path)

        n = len(arxiv_data)
        ai_writing_FD = [None] * n

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:

            future_to_key = {}
            for tag, strategy_fn in strategies:
                for i in range(n):
                    fut = executor.submit(
                        strategy_fn,
                        openai_oss_query,
                        arxiv_data[abstract_col].iloc[i],
                        model_name,
                    )
                    future_to_key[fut] = (tag, i)

            iterator = tqdm(
                concurrent.futures.as_completed(future_to_key),
                total=n * len(strategies),
                desc="Generating new abstracts",
            )

            for fut in iterator:
                tag, idx = future_to_key[fut]
                result = fut.result()
                ai_writing_FD[idx] = result[0]

        arxiv_data[rewrite_col] = ai_writing_FD
        # import pdb; pdb.set_trace()
        arxiv_data.to_parquet(f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_fd_{category}_{subsample_size//2}_fronthalf.parquet")

        print(f"saved for year {year}")
