import os
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import pandas as pd
from oss_api import openai_oss_query
from gemini_api import call_gemini_2, call_gemini_3
from llama_api import llama_query

query_fns = [llama_query, call_gemini_3, openai_oss_query, call_gemini_2]
llm_labels = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
assert(len(query_fns) == len(llm_labels))


def rewrite_abstract(prompt_model, abstract, model_name):

    context1 = """
    The aim here is to reverse-engineer the author's writing process by
    compressing a piece of text into a concise structured form.
    First summarize the goal of the text (introduction, method, results, etc.)
    and then reverse-engineer it into bullet points.
    """

    prompt1 = f"Here is the text: {abstract}"
    res1 = prompt_model(context1, prompt1)

    context2 = """
    Expand the concise bullet-point version into a full abstract.
    """

    prompt2 = f"Here is the writing: {res1}"
    res2 = prompt_model(context2, prompt2)

    context3 = """
    Proofread the writing for grammatical accuracy.
    Return ONLY the corrected abstract with no titles or extra text.
    """

    prompt3 = f"Here is the writing: {res2}"
    res3 = prompt_model(context3, prompt3)

    return res3, model_name


if __name__ == "__main__":

    subsample_size = 20000
    to_rewrite = 100
    category = "cs."

    arxiv_path = f"/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_{category}_{subsample_size}.json"

    with open(arxiv_path, 'rb') as f:
        arxiv_data = json.load(f)

    train_years = ['2010']

    cols = ["human_abstract"] + llm_labels
    data_df = pd.DataFrame(columns=cols)

    for year in tqdm(train_years):

        abstracts = arxiv_data[year][:to_rewrite]

        # results[abstract_idx][model_name] = rewrite
        results = defaultdict(dict)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:

            future_to_key = {}

            for i, abstract in enumerate(abstracts):
                for model_idx, (query_fn, label) in enumerate(zip(query_fns, llm_labels)):

                    future = executor.submit(
                        rewrite_abstract,
                        query_fn,
                        abstract,
                        label
                    )

                    future_to_key[future] = (i, label)

            iterator = tqdm(
                concurrent.futures.as_completed(future_to_key),
                total=len(future_to_key),
                desc="Generating new abstracts",
            )

            for fut in iterator:
                i, label = future_to_key[fut]
                rewrite, _ = fut.result()
                results[i][label] = rewrite

        rows = []

        for i, abstract in enumerate(abstracts):

            row = {"human_abstract": abstract}

            for label in llm_labels:
                row[label] = results[i].get(label)

            rows.append(row)

        year_df = pd.DataFrame(rows)

        # import pdb; pdb.set_trace()

        year_df.to_parquet(f"other_4_{to_rewrite}_{train_years[0]}.parquet")

        print(f"saved for year {year}")