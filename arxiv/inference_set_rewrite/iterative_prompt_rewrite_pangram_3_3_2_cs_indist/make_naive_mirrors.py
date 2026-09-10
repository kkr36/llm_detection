"""Step 4: mirror the train/val human abstracts with the DEFAULT/naive prompt (rewrite_abstract
from inference_set_rewrite/rewrite.py) and GPT-OSS-120b. These naive mirrors (mirror_0) are the
source AI abstracts the agent is meant to humanize.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from openai_api import openai_oss_query

LLM_LABEL = "GPT OSS 120b"


def rewrite_abstract(prompt_model, abstract, model_name):
    """Naive default mirror: reverse-engineer -> expand -> proofread (verbatim from rewrite.py)."""
    context1 = """
    The aim here is to reverse - engineer the author 's writing process by taking a piece of text from a paper and compressing it into a more
    concise form. This process simulates how an author might distill
    their thoughts and key points into a structured, yet not overly
    condensed form.
    Now as a first step, first summarize the goal of the text , e.g., is it
    introduction, or method, results? and then given a complete piece of
    text from a paper, reverse-engineer it into a list of bullet points.
    """
    res1 = prompt_model(context1, f"Here is the text: {abstract}")

    context2 = """
    Following the initial step of reverse-engineering the author's writing
    process by compressing a text segment from a paper, you now enter the
    second phase. Here, your objective is to expand upon the concise
    version previously crafted . This stage simulates how an author
    elaborates on the distilled thoughts and key points, enriching them
    into a detailed, structured narrative.
    Given the concise output from the previous step, your task is to develop
    it into a fully fleshed-out text (abstract, specifically).
    """
    res2 = prompt_model(context2, f"Here is the writing: {res1}")

    context3 = """
    Your task is to proofread the provided writing for grammatical accuracy.
    Ensure that the corrections introduce minimal distortion to the
    original content. Return only the corrected abstract, without ANY fluff or titles at the start.
    """
    res3 = prompt_model(context3, f"Here is the writing: {res2}")
    return res3, model_name


def mirror_file(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    abstracts = df["human_abstract"].tolist()
    mirrors = [None] * len(abstracts)

    def work(i):
        m, _ = rewrite_abstract(openai_oss_query, abstracts[i], LLM_LABEL)
        return i, m

    print(f"[{in_csv}] naive-mirroring {len(abstracts)} abstracts ...")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(work, i) for i in range(len(abstracts))]
        for fut in tqdm(as_completed(futs), total=len(abstracts)):
            i, m = fut.result()
            mirrors[i] = m

    out = pd.DataFrame({
        "arxiv_id": df["arxiv_id"],
        "categories": df["categories"],
        "original": df["human_abstract"],   # the human source
        "mirroring_llm": LLM_LABEL,
        "mirror_0": mirrors,                 # the naive AI mirror (source to humanize)
    })
    out.to_csv(out_csv, index=False)
    print(f"saved {out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    mirror_file("abstracts_train_75.csv", "results_0_oss_train_75.csv")
    mirror_file("abstracts_val_25.csv", "results_0_oss_val_25.csv")
