"""
Sample 500 ADDITIONAL 2020 human abstracts, disjoint from the 100 already analyzed
(2020_human_sample.csv), sentence-split them, and write ../data/2020_train.csv.

Non-overlap is guaranteed by reproducing the exact original sample (seed 42, same
notna filter) and dropping those rows before sampling the new 500.
"""
import os, sys
import pandas as pd

# Torch-free sentence splitter: replicate data_helper.split_into_sentences with a
# direct spaCy import so this runs on any CPU node (importing data_helper pulls in
# torch, which SIGILLs on older CPU nodes).
import en_core_web_lg
_nlp = en_core_web_lg.load(disable=["ner", "parser"])
_nlp.enable_pipe("senter")


def split_into_sentences(abstracts, labels, batch_size=200):
    all_sentences, all_labels = [], []
    for doc, label in zip(_nlp.pipe(abstracts, batch_size=batch_size), labels):
        sentences = [sent.text.strip() for sent in doc.sents]
        all_sentences.extend(sentences)
        all_labels.extend([label] * len(sentences))
    return all_sentences, all_labels

P2020 = "/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_ai_cs._10000_fronthalf_120b_qwen_codex.parquet"
OUT = "/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis/data"
SEED = 42
N_ANALYZED = 100   # the abstracts already scored in pipeline.py
N_TRAIN = 500


def prefix_key(text, n=100):
    return " ".join(str(text).split())[:n]


df = pd.read_parquet(P2020)
pool = df[df["human_abstract"].notna()]

# reproduce the exact 100 already analyzed, then exclude them
analyzed = pool.sample(n=N_ANALYZED, random_state=SEED)
remaining = pool.drop(analyzed.index)
train = remaining.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)

# hard check: no overlap with the analyzed sample already on disk
prev = pd.read_csv(f"{OUT}/2020_human_sample.csv")["prefix_key"].tolist()
train_keys = train["human_abstract"].map(prefix_key)
overlap = set(train_keys) & set(prev)
assert not overlap, f"overlap with analyzed sample: {len(overlap)}"
print(f"[make_2020_train] sampled {len(train)} abstracts, 0 overlap with the 100 analyzed")

# sentence-split, keeping abstract membership
abstracts = train["human_abstract"].tolist()
rows = []
for ai, ab in enumerate(abstracts):
    ss, _ = split_into_sentences([ab], [0])
    ss = [s for s in ss if s.strip()] or [ab.strip()]
    for si, s in enumerate(ss):
        rows.append({
            "year": 2020,
            "abstract_id": f"2020train_{ai}",
            "abstract_local_idx": ai,
            "sentence_idx": si,
            "sentence": s,
            "prefix_key": prefix_key(ab),
        })

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/2020_train.csv", index=False)
print(f"[make_2020_train] {len(abstracts)} abstracts -> {len(out)} sentences "
      f"-> {OUT}/2020_train.csv")
