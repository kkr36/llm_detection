import os
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import pandas as pd
from llama_api import llama_query
from openai_api import openai_oss_query
from gemini_api import call_gemini_2_flash, call_gemini_2_flash_lite, call_gemini_2_pro

def rewrite_abstract(prompt_model, abstract, model_name):

    context1 = f"""
    The aim here is to reverse - engineer the author 's writing process by taking a piece of text from a paper and compressing it into a more
    concise form. This process simulates how an author might distill
    their thoughts and key points into a structured, yet not overly
    condensed form.
    Now as a first step, first summarize the goal of the text , e.g., is it
    introduction, or method, results? and then given a complete piece of
    text from a paper, reverse-engineer it into a list of bullet points.
    """
    prompt1 = f"Here is the text: {abstract}"
    res1 = prompt_model(context1, prompt1)

    context2 = f"""
    Following the initial step of reverse-engineering the author's writing
    process by compressing a text segment from a paper, you now enter the
    second phase. Here, your objective is to expand upon the concise
    version previously crafted . This stage simulates how an author
    elaborates on the distilled thoughts and key points, enriching them
    into a detailed, structured narrative.
    Given the concise output from the previous step, your task is to develop
    it into a fully fleshed-out text (abstract, specifically).
    """
    prompt2 = f"Here is the writing: {res1}"
    res2 = prompt_model(context2, prompt2)

    context3 = f"""
    Your task is to proofread the provided writing for grammatical accuracy.
    Ensure that the corrections introduce minimal distortion to the
    original content. Return only the corrected abstract, without ANY fluff or titles at the start.
    """
    prompt3 = f"Here is the writing: {res2}"
    res3 = prompt_model(context3, prompt3)
    return res3, model_name

if __name__ == "__main__":
    subsample_size = 20000
    category = "cs."
    arxiv_path = f"/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_{category}_{subsample_size}.json"
    with open(arxiv_path, 'rb') as f:
        arxiv_data = json.load(f)

    train_years = ['2020']
    print(train_years)
    
    existing_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_ai_{category}_{subsample_size//2}_fronthalf_gemini.parquet"

    data_df = pd.read_parquet(existing_path)

    # sample 2500 new abstracts that weren't included in data_df['human_abstract']

    import random
    random.seed(42) 
    existing_abstracts = set(data_df['human_abstract'].dropna())

    new_abstracts = [
        abstract for abstract in arxiv_data['2020']
        if abstract not in existing_abstracts
    ]

    print(f"Found {len(new_abstracts)} new abstracts in arxiv_data['2020']")

    # Sample 2500 if we have enough, otherwise take all available
    if len(new_abstracts) >= 2500:
        sampled_abstracts = random.sample(new_abstracts, 2500)
    else:
        print(f"Warning: Only {len(new_abstracts)} new abstracts available")
        sampled_abstracts = new_abstracts

    # sampled_abstracts = sampled_abstracts[:10]

    ai_writing = [None] * len(sampled_abstracts)

    # import pdb; pdb.set_trace()

    # sample 2500 abstracts that were not included in the original sample

    # for each abstact, rewrite with pro

    # resave

    # import pdb; pdb.set_trace()

    # iterator = tqdm(
    #     range(len(arxiv_data[year])),
    #     total=len(arxiv_data[year]),
    #     desc="Generating new abstracts",
    # )

    # for i in iterator:
    #     ai_writing[i] = rewrite_abstract(
    #         query_fns[llms[i]],        # prompt_model
    #         arxiv_data[year][i],       # abstract
    #         llm_labels[llms[i]]        # model_name
    #     )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

        future_to_idx = {
            executor.submit(
                rewrite_abstract,
                call_gemini_2_pro,        # prompt_model
                sampled_abstracts[i],       # abstract
                "Gemini 2.5 Pro"        # model_name
            ): i
            for i in range(len(sampled_abstracts))
        }

        iterator = tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(sampled_abstracts),
            desc="Generating new abstracts",
        )

        for fut in iterator:
            idx = future_to_idx[fut]
            ai_writing[idx] = fut.result()

    # import pdb; pdb.set_trace()
    rows = []
    for i, (rewrite, model_name) in enumerate(ai_writing):
        row = {
            "human_abstract": sampled_abstracts[i],
            # Place rewrite under the correct model column
            model_name: rewrite
        }
        rows.append(row)
    # import pdb; pdb.set_trace()

    # Convert to DataFrame
    year_df = pd.DataFrame(rows)
    # Concatenate the dataframes
    data_df = pd.concat([data_df, year_df], ignore_index=True).reset_index(drop=True)

    # Fill NaN values with None if you prefer (optional)
    # data_df = data_df.fillna(None)

    # import pdb; pdb.set_trace()

    data_df.to_parquet(f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_ai_{category}_{subsample_size//2}_fronthalf_gemini_full.parquet")
        