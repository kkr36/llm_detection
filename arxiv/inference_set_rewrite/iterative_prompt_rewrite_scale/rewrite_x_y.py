import os
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import pandas as pd
from openai_api import openai_oss_query
from strategy import rewrite_strategy_X, rewrite_strategy_Y

model_name = "GPT OSS 120b"

if __name__ == "__main__":
    subsample_size = 20000
    category = "cs."
    arxiv_path = f"/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_{category}_{subsample_size}.json"
    with open(arxiv_path, 'rb') as f:
        arxiv_data = json.load(f)

    train_years = [str(x) for x in range(2010,2021,2)][-1:]
    print(train_years)

    for year in tqdm(train_years):
        arxiv_data[year] = arxiv_data[year][:subsample_size//2] # TODO remove for full sample (will need to merge with subsample_size//2: first)

        n = len(arxiv_data[year])
        ai_writing_X = [None] * n
        ai_writing_Y = [None] * n

        strategies = [
            ("X", rewrite_strategy_X, ai_writing_X),
            ("Y", rewrite_strategy_Y, ai_writing_Y),
        ]

        # import pdb; pdb.set_trace()

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:

            future_to_key = {}
            for tag, strategy_fn, _ in strategies:
                for i in range(n):
                    fut = executor.submit(
                        strategy_fn,
                        openai_oss_query,
                        arxiv_data[year][i],
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
                if tag == "X":
                    ai_writing_X[idx] = result[0]
                else:
                    ai_writing_Y[idx] = result[0]

        rows = []
        for i in range(n):
            row = {
                "human_abstract": arxiv_data[year][i],
                "rewrite_X": ai_writing_X[i],
                "rewrite_Y": ai_writing_Y[i],
            }
            rows.append(row)

        # Convert to DataFrame
        year_df = pd.DataFrame(rows)
        # import pdb; pdb.set_trace()
        year_df.to_parquet(f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_xy_{category}_{subsample_size//2}_fronthalf.parquet")
        # year_df.to_csv("test.csv")
        # year_df.to_parquet("test.parquet")
        
        print(f"saved for year {year}")
    