#!/home/kkr36/.conda/envs/llm_embeddings/bin/python
"""
Embed randomly sampled sentences from human and AI sources for multiple years,
saving each year as separate keys in a single .npz file.

Sources and labels:
  0 = Human
  1 = GPT OSS 120b
  2 = Llama 3.3 70b Instruct
  3 = Gemini 3 Preview
  4 = Qwen

Output: embeddings_years.npz with keys like 'embeddings_2010', 'labels_2010', etc.
"""

import json
import random

import en_core_web_lg
import numpy as np
import pandas as pd
from google import genai
from tqdm import tqdm

DATA_DIR = "/share/garg/arxiv_kaggle/multillm/data_raw/"
OUT_PATH = "/home/kkr36/llm_detection/arxiv/separation/embeddings_years.npz"
YEARS = [2010, 2012, 2014, 2016, 2018, 2020]
SENTENCES_PER_CLASS = 5000
SEED = 42
MODEL_ID = "gemini-embedding-2"
BATCH_SIZE = 100

SOURCES = [
    ("human_abstract",         0),
    ("GPT OSS 120b",           1),
    ("Llama 3.3 70b Instruct", 2),
    ("Gemini 3 Preview",       3),
    ("Qwen",                   4),
]

nlp = en_core_web_lg.load(disable=["ner", "parser"])
nlp.enable_pipe("senter")


def split_into_sentences(texts):
    sentences = []
    for doc in tqdm(nlp.pipe(texts, batch_size=200), total=len(texts), desc="  splitting"):
        sentences.extend(sent.text.strip() for sent in doc.sents)
    return sentences


def embed_batch(texts, client, max_retries=5, base_delay=3.0):
    import time
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(model=MODEL_ID, contents=texts)
            return np.array([e.values for e in result.embeddings])
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"  API error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)


def embed_year(year, df, client):
    rng = random.Random(SEED)
    all_sentences, all_labels = [], []

    for col, label in SOURCES:
        name = col if col != "human_abstract" else "Human"
        print(f"  [{label}] {name}")
        texts = df[col].dropna().tolist()
        sents = split_into_sentences(texts)
        sents = [s for s in sents if s.strip()]

        if len(sents) < SENTENCES_PER_CLASS:
            print(f"    Warning: only {len(sents)} sentences available, using all.")
            sampled = sents
        else:
            sampled = rng.sample(sents, SENTENCES_PER_CLASS)

        print(f"    {len(sampled)} sampled from {len(sents)} total")
        all_sentences.extend(sampled)
        all_labels.extend([label] * len(sampled))

    print(f"  Total sentences to embed: {len(all_sentences)}")
    batches = [
        all_sentences[i : i + BATCH_SIZE]
        for i in range(0, len(all_sentences), BATCH_SIZE)
    ]

    all_embeddings = []
    for batch in tqdm(batches, desc=f"  Embedding {year}"):
        all_embeddings.append(embed_batch(batch, client))

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels, dtype=np.int32)
    return embeddings, labels


def main():
    np.random.seed(SEED)

    with open("/home/kkr36/creds.json") as f:
        keys = json.load(f)
    client = genai.Client(api_key=keys["gemini_api_key"])

    results = {}
    for year in YEARS:
        data_path = f"{DATA_DIR}arxiv_{year}_ai_cs._10000_fronthalf_120b_qwen.parquet"
        print(f"\n=== Year {year} ===")
        df = pd.read_parquet(data_path)
        embeddings, labels = embed_year(year, df, client)
        results[f"embeddings_{year}"] = embeddings
        results[f"labels_{year}"] = labels
        print(f"  Embeddings shape: {embeddings.shape}")

    np.savez(OUT_PATH, **results)
    print(f"\nSaved to {OUT_PATH}")
    for year in YEARS:
        emb = results[f"embeddings_{year}"]
        lbl = results[f"labels_{year}"]
        print(f"  {year}: {emb.shape}, {lbl.shape}")


if __name__ == "__main__":
    main()
