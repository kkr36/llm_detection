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

query_fns = [llama_query, call_gemini_3, openai_oss_query, call_gemini_2]
# query_fns = [openai_oss_query] * 4
llm_labels = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
assert(len(query_fns) == len(llm_labels))

import pickle; data = pickle.load(open("my_data.pkl", "rb"))

def rewrite_abstract_oneshot(prompt_model, abstract, model_name):

    context1 = """
    You are rewriting scientific paper abstracts. When given an abstract, produce a rewritten version that:
    - Preserves all findings, methods, and conclusions
    - Matches the original word count closely
    - Uses fresh sentence structures and phrasing throughout — avoid mirroring the original sentence by sentence
    - Maintains an ArXiv pre-print tone
    Output only the rewritten abstract, no preamble or commentary.
    """
    prompt1 = f"""
    Rewrite this abstract:

    {abstract}
    """

    res1 = prompt_model(context1, prompt1)

    context2 = """
    You are a scientific copy-editor. You will be given a passage and a reference text.
    Your tasks:
    1. Remove any meta-commentary (e.g. "Here is a rewritten version...") from the start or end of the passage.
    2. If any phrases or sentences are lifted too closely from the reference text, rephrase them in the style of the surrounding passage.
    3. Fix grammatical errors, keeping changes minimal.
    Output only the cleaned passage, no commentary.
    """

    prompt2 = f"""
    Passage:
    {res1}

    Reference Text:
    {abstract}
    """

    res2 = prompt_model(context2, prompt2)

    return res1, res2, model_name

def rewrite_abstract_autocomplete(prompt_model, abstract, model_name):
    first_sentences, _ = split_sentences(abstract)

    # prompt1 = f"""
    # Write a scientific paper abstract (style of an ArXiv pre-print). Start with these sentences : {first_sentences}. Make the article (minus the provided sentences) about {len(abstract.split())} words long .
    # """

    prompt1 = f"""
    Write a scientific paper abstract (style of an ArXiv pre-print). The abstract has already begun with the following sentences — do not repeat or include them in your response, only continue from where they leave off: "{first_sentences}". Write approximately {len(abstract.split())} words.
    """
    res1 = prompt_model('', prompt1)

    prompt2 = f"""
    Below is a passage and a prefix. Your task is to return a cleaned-up version of the passage:
    1. If the passage begins with the prefix (or a close paraphrase of it), remove it.
    2. Fix any grammatical errors in the remaining text. Ensure that the corrections introduce minimal distortion to the original content.
    3. Return only the cleaned text, with no commentary.

    Prefix:
    {first_sentences}

    Passage:
    {res1}
    """

    res2 = prompt_model('', prompt2)

    return res1, res2, model_name

def rewrite_abstract_length_match(prompt_model, abstract, model_name):
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
    version previously crafted. This stage simulates how an author
    elaborates on the distilled thoughts and key points, enriching them
    into a detailed, structured narrative.
    Given the concise output from the previous step, your task is to develop
    it into a fully fleshed-out text (abstract, specifically). Make the abstract around {len(abstract.split())} words long.
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
    # arxiv_path = f"/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_{category}_{subsample_size}.json"
    # with open(arxiv_path, 'rb') as f:
    #     arxiv_data = json.load(f)
    # import pdb; pdb.set_trace()

    # train_years = [str(x) for x in range(2013,2026,1)]
    # train_years = ['2013', '2018', '2020', '2023', '2025']
    # train_years = ['2020', '2023', '2025']
    # train_years = [str(x) for x in range(2013, 2018)]
    train_years = [str(x) for x in range(2010,2020,2)][:1]
    print(train_years)

    cols = ["human_abstract"] + llm_labels
    data_df = pd.DataFrame(columns=cols)

    old_text = []
    new_text = []
    human_text = []
    interm_text = []

    to_rewrite = 20
    val_start = 2500
    test_start = 2600

    for year in tqdm(train_years):
      fpath = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size//2}_fronthalf.parquet"
      arxiv_data = pd.read_parquet(fpath)
      llms = [i % len(llm_labels) for i in range(len(arxiv_data))]
      changed = False

      for i in tqdm(range(val_start, val_start + to_rewrite)):
          row = arxiv_data.iloc[i]
          orig_abs = row['human_abstract']
          llm_abs = row[llm_labels[llms[i]]]
          assert llm_abs is not None and len(llm_abs) > 0

          inter_abs, new_abs, _ = rewrite_abstract_oneshot(query_fns[llms[i]], orig_abs, llm_labels[llms[i]])

          old_text.append(llm_abs)
          new_text.append(new_abs)
          human_text.append(orig_abs)
          interm_text.append(inter_abs)
    
    import pdb; pdb.set_trace()
    print("hi")
    