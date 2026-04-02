import os
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import pandas as pd
from openai_api import openai_oss_query
from strategy import rewrite_strategy_Z

model_name = "GPT OSS 120b"

if __name__ == "__main__":
    subsample_size = 20000
    category = "cs."
    train_years = [2020]
    strategies = [("Z", rewrite_strategy_Z)]

    abstract_col = "human_abstract"

    for year in tqdm(train_years):

        arxiv_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_xy_{category}_{subsample_size//2}_fronthalf.parquet"
        arxiv_data = pd.read_parquet(arxiv_path)

        n = len(arxiv_data)
        ai_writing_Z = [None] * n

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
                ai_writing_Z[idx] = result[0]

        arxiv_data["rewrite_Z"] = ai_writing_Z
        # import pdb; pdb.set_trace()
        arxiv_data.to_parquet(f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_xyz_{category}_{subsample_size//2}_fronthalf.parquet")

        print(f"saved for year {year}")
