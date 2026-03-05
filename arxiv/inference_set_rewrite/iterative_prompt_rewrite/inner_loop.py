import os
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import pandas as pd
from llama_api import llama_query
from openai_api import openai_oss_query
from gemini_api import call_gemini_2, call_gemini_3
from util import *
from strategy import CURRENT_STRATEGY, CURRENT_TIMESTEP

query_fns = [call_gemini_3, openai_oss_query]
# query_fns = [openai_oss_query] * 4
llm_labels = ["Gemini 3 Preview", "GPT OSS 120b"]
assert(len(query_fns) == len(llm_labels))

if __name__ == "__main__":
    subsample_size = 20000
    category = "cs."
    train_year = 2010

    cols = ["human_abstract"] + llm_labels
    data_df = pd.DataFrame(columns=cols)

    old_text = []
    new_text = []
    human_text = []
    interm_text = []

    to_rewrite = 20
    val_start = 2500
    test_start = 2600

    fpath = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{train_year}_ai_{category}_{subsample_size//2}_fronthalf.parquet"
    arxiv_data = pd.read_parquet(fpath)
    llms = [i % len(llm_labels) for i in range(len(arxiv_data))]
    changed = False

    for i in tqdm(range(val_start, val_start + to_rewrite)):
        row = arxiv_data.iloc[i]
        orig_abs = row['human_abstract']
        llm_abs = row[llm_labels[llms[i]]]
        assert llm_abs is not None and len(llm_abs) > 0

        new_abs, _ = CURRENT_STRATEGY(query_fns[llms[i]], orig_abs, llm_labels[llms[i]])

        old_text.append(llm_abs)
        new_text.append(new_abs)
        human_text.append(orig_abs)
        interm_text.append(inter_abs)
    
    import pdb; pdb.set_trace()
    print("hi")
    