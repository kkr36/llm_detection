#!/home/kkr36/.conda/envs/llm_embeddings/bin/python
"""
Embed randomly sampled sentences from human and Gemini model sources.

Sources and labels:
  0 = Human
  1 = Gemini 2.0 Flash-Lite
  2 = Gemini 3 Preview
  3 = Gemini 2.0 Flash
  4 = Gemini 2.5 Flash
  5 = Gemini 2.5 Pro

Output: embeddings_gemini_full.npz with keys 'embeddings' (N x D) and 'labels' (N,)
"""

import json
import random

import en_core_web_lg
import numpy as np
from google import genai
from tqdm import tqdm

DATA_PATH = (
    "/share/garg/arxiv_kaggle/multillm/data_raw/"
    "arxiv_2020_ai_cs._10000_fronthalf_gemini_full.parquet"
)
OUT_PATH = "/home/kkr36/llm_detection/arxiv/separation/embeddings_gemini_full.npz"
SENTENCES_PER_CLASS = 5000
SEED = 42
MODEL_ID = "gemini-embedding-2"
BATCH_SIZE = 100

SOURCES = [
    ("human_abstract",    0),
    ("Gemini 2.0 Flash-Lite", 1),
    ("Gemini 3 Preview",  2),
    ("Gemini 2.0 Flash",  3),
    ("Gemini 2.5 Flash",  4),
    ("Gemini 2.5 Pro",    5),
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


def main():
    import pandas as pd

    rng = random.Random(SEED)
    np.random.seed(SEED)

    df = pd.read_parquet(DATA_PATH)

    with open("/home/kkr36/creds.json") as f:
        keys = json.load(f)
    client = genai.Client(api_key=keys["gemini_api_key"])

    all_sentences, all_labels = [], []

    for col, label in SOURCES:
        name = col if col != "human_abstract" else "Human"
        print(f"\n[{label}] {name}")
        texts = df[col].dropna().tolist()
        sents = split_into_sentences(texts)
        sents = [s for s in sents if s.strip()]

        if len(sents) < SENTENCES_PER_CLASS:
            print(f"  Warning: only {len(sents)} sentences available, using all.")
            sampled = sents
        else:
            sampled = rng.sample(sents, SENTENCES_PER_CLASS)

        print(f"  {len(sampled)} sampled from {len(sents)} total sentences")
        all_sentences.extend(sampled)
        all_labels.extend([label] * len(sampled))

    print(f"\nTotal sentences to embed: {len(all_sentences)}")

    batches = [
        all_sentences[i : i + BATCH_SIZE]
        for i in range(0, len(all_sentences), BATCH_SIZE)
    ]

    all_embeddings = []
    for batch in tqdm(batches, desc="Embedding"):
        all_embeddings.append(embed_batch(batch, client))

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels, dtype=np.int32)

    np.savez(OUT_PATH, embeddings=embeddings, labels=labels)
    print(f"\nSaved to {OUT_PATH}")
    print(f"Embeddings shape: {embeddings.shape}")
    for col, lbl in SOURCES:
        n = (labels == lbl).sum()
        print(f"  label {lbl} ({col}): {n} sentences")


if __name__ == "__main__":
    main()
