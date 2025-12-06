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
    subsample_size = 20000
    category = "cs."
    arxiv_path = f"/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_{category}_{subsample_size}.json"
    with open(arxiv_path, 'rb') as f:
        arxiv_data = json.load(f)
    import pdb; pdb.set_trace()
    # train_years = [str(x) for x in range(2013,2026,1)]
    # train_years = ['2013', '2018', '2020', '2023', '2025']
    # train_years = ['2020', '2023', '2025']
    # train_years = [str(x) for x in range(2013, 2018)]
    train_years = [str(x) for x in range(2010,2020,2)]
    print(train_years)

    cols = ["human_abstract"] + llm_labels
    data_df = pd.DataFrame(columns=cols)

    for year in tqdm(train_years):
        arxiv_data[year] = arxiv_data[year][:subsample_size//2] # TODO remove for full sample (will need to merge with subsample_size//2: first)
        # arxiv_data[year] = arxiv_data[year]
        # import pdb; pdb.set_trace()

        llms = [i % len(llm_labels) for i in range(len(arxiv_data[year]))]

        ai_writing = [None] * len(arxiv_data[year])
        # tmp = rewrite_abstract(openai_oss_query, arxiv_data[year][1], "gemini 25 pro")
        # import pdb; pdb.set_trace()

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:

            future_to_idx = {
                executor.submit(
                    rewrite_abstract,
                    query_fns[llms[i]],        # prompt_model
                    arxiv_data[year][i],       # abstract
                    llm_labels[llms[i]]        # model_name
                ): i
                for i in range(len(arxiv_data[year]))
            }

            iterator = tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(arxiv_data[year]),
                desc="Generating new abstracts",
            )

            for fut in iterator:
                idx = future_to_idx[fut]
                ai_writing[idx] = fut.result()

        # import pdb; pdb.set_trace()
        rows = []
        for i, (rewrite, model_name) in enumerate(ai_writing):
            row = {
                "human_abstract": arxiv_data[year][i],
                # Place rewrite under the correct model column
                model_name: rewrite
            }
            rows.append(row)

        # Convert to DataFrame
        year_df = pd.DataFrame(rows)
        # import pdb; pdb.set_trace()
        year_df.to_parquet(f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size//2}_fronthalf.parquet")
        # year_df.to_csv("test.csv")
        # year_df.to_parquet("test.parquet")
        
        print(f"saved for year {year}")
    