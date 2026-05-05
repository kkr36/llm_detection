import os
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import pandas as pd
from llama_api import llama_query
from openai_api import openai_oss_120b_query
from gemini_api import call_gemini_2, call_gemini_3
from rewrite import rewrite_abstract

query_fns = [llama_query, call_gemini_3, openai_oss_120b_query, call_gemini_2]
llm_labels = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
assert(len(query_fns) == len(llm_labels))

if __name__ == "__main__":
    subsample_size = 20000//2
    rewrite_pct = 0.2
    category = "cs."
    train_years = [str(x) for x in range(2010,2021,2)][-1:]
    print(train_years)

    for year in tqdm(train_years):

        arxiv_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size}_fronthalf_120b.parquet"
        arxiv_data = pd.read_parquet(arxiv_path)
        assert(len(arxiv_data) == subsample_size)
        num_rewrite = int(len(arxiv_data) * rewrite_pct)

        llms_old = [i % len(llm_labels) for i in range(num_rewrite)]
        llms_new = [(i+1) % len(llm_labels) for i in range(num_rewrite)]

        # only rewrite rows assigned to openai_oss_120b_query (llms_new == 2)
        oss_120b_idx = [i for i in range(num_rewrite) if llms_new[i] == 2]

        ai_writing = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:

            future_to_idx = {
                executor.submit(
                    rewrite_abstract,
                    openai_oss_120b_query,
                    arxiv_data.iloc[i][llm_labels[llms_old[i]]],
                    llm_labels[2]
                ): i
                for i in oss_120b_idx
            }

            iterator = tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(oss_120b_idx),
                desc="Generating new abstracts",
            )

            for fut in iterator:
                idx = future_to_idx[fut]
                ai_writing[idx] = fut.result()

        for i, (rewrite, model_name) in ai_writing.items():
            arxiv_data.at[i, model_name] = rewrite

        arxiv_data.to_parquet(f"/share/garg/arxiv_kaggle/multillm/double_rewrite/arxiv_{year}_ai_{category}_{subsample_size}_{rewrite_pct}_fronthalf_120b.parquet")

        print(f"saved for year {year}")
