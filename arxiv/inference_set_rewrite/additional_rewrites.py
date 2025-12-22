# for each year:
#   randomly subsample 20% of abstracts
#   rewrite the llm abstract with a new llm (make it a diff one)
#   add to the df of abstracts
#   save df under new filename

import os
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import pandas as pd
from llama_api import llama_query
from openai_api import openai_oss_query
from gemini_api import call_gemini_2, call_gemini_3

query_fns = [llama_query, call_gemini_3, openai_oss_query, call_gemini_2]
# query_fns = [openai_oss_query] * 4
llm_labels = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
assert(len(query_fns) == len(llm_labels))

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
    subsample_size = 20000//2
    rewrite_pct = 0.2
    category = "cs."
    # arxiv_path = f"/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_{category}_{subsample_size}.json"
    # with open(arxiv_path, 'rb') as f:
    #     arxiv_data = json.load(f)
    # train_years = [str(x) for x in range(2013,2026,1)]
    # train_years = ['2013', '2018', '2020', '2023', '2025']
    # train_years = ['2020', '2023', '2025']
    # train_years = [str(x) for x in range(2013, 2018)]
    train_years = [str(x) for x in range(2010,2021,2)]
    print(train_years)

    for year in tqdm(train_years):

        arxiv_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size}_fronthalf.parquet"
        arxiv_data = pd.read_parquet(arxiv_path)
        assert(len(arxiv_data) == subsample_size)
        num_rewrite = int(len(arxiv_data) * rewrite_pct)
        
        llms_old = [i % len(llm_labels) for i in range(num_rewrite)]
        llms_new = [(i+1) % len(llm_labels) for i in range(num_rewrite)] # TODO should we keep it as +1? or randomly select a second llm

        ai_writing = [None] * num_rewrite

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:

            future_to_idx = {
                executor.submit(
                    rewrite_abstract,
                    query_fns[llms_new[i]],        # prompt_model
                    arxiv_data.iloc[i][llm_labels[llms_old[i]]],       # abstract
                    llm_labels[llms_new[i]]        # model_name
                ): i
                for i in range(num_rewrite)
            }

            iterator = tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=num_rewrite,
                desc="Generating new abstracts",
            )

            for fut in iterator:
                idx = future_to_idx[fut]
                ai_writing[idx] = fut.result()

        # rows = []
        for i, (rewrite, model_name) in enumerate(ai_writing):
            arxiv_data.at[i, model_name] = rewrite
            # row = {
            #     "human_abstract": arxiv_data[year][i],
            #     # Place rewrite under the correct model column
            #     model_name: rewrite
            # }
            # rows.append(row)

        # Convert to DataFrame
        # year_df = pd.DataFrame(rows)
        # import pdb; pdb.set_trace()
        arxiv_data.to_parquet(f"/share/garg/arxiv_kaggle/multillm/double_rewrite/arxiv_{year}_ai_{category}_{subsample_size}_{rewrite_pct}_fronthalf.parquet")
        # year_df.to_csv("test.csv")
        # year_df.to_parquet("test.parquet")
        
        print(f"saved for year {year}")
    