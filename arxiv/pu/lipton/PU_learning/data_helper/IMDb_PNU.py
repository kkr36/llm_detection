import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import re

from .IMDb import (
    split_into_sentences,
    initialize_bert_transform,
    _is_xz_col, _is_xz_float_col, _is_xz_count_col, _parse_xz_counts,
    _interleave_xz_cols_counts, _get_human_sentences,
    _interleave_xz_cols, _interleave_xz_cols_frac,
)

# Label convention (same as IMDb.py, plus label=2 for labeled negatives):
#   label=1 -> confirmed/labeled positive  -> p_data
#   label=2 -> confirmed/labeled negative  -> ln_data  (NEW)
#   label=0 -> unlabeled pool (mix)        -> n_data


class IMDbBERTData_PNU(torch.utils.data.Dataset):
    """Like IMDbBERTData but supports a third label=2 category for labeled negatives."""
    def __init__(self, data, labels, transform):
        labels = np.array(labels)
        encodings = transform(data)

        p_data_idx  = np.where(labels == 1)[0]
        ln_data_idx = np.where(labels == 2)[0]
        n_data_idx  = np.where(labels == 0)[0]

        self.p_data  = encodings[p_data_idx,  :, :]
        self.ln_data = encodings[ln_data_idx, :, :]  # labeled negatives
        self.n_data  = encodings[n_data_idx,  :, :]  # unlabeled pool

        self.labels = labels
        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.labels)


def read_arxiv_split2_PNU_pos_2010(split_dir, alpha, split, sentence, inject, seed, n_labeled=0):
    """
    PNU version of read_arxiv_split2.
    Source domain: 2010 writing (fixed path); labeled positives (ai) and negatives (human).
    Target domain: test-year writing (split_dir); treated as fully unlabeled.
    split=='train': source labeled pos (rows 0-1500 ai) + labeled neg (rows 2000-8000 human)
                    + target unlabeled (rows 1500-6000 split by alpha).
    split=='val':   source labeled pos only (rows 1500-2000 ai)
                    + target unlabeled (rows 6500-8000 split by alpha).
    """
    assert seed is not None
    inject = False

    SOURCE_2010_PATH = '/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2010_ai_cs._10000_fronthalf_120b_qwen.parquet'
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen"]

    # Load and prepare source (2010) data
    source_df = pd.read_parquet(SOURCE_2010_PATH)
    source_llm_writing = []
    for i in tqdm(list(range(len(source_df)))):
        assert len(source_df.iloc[i][llm_cols[i % 4]]) > 0
        source_llm_writing.append(source_df.iloc[i][llm_cols[i % 4]])
    source_df['ai_abstract'] = source_llm_writing
    source_df = source_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Load and prepare target (test year) data
    target_df = pd.read_parquet(split_dir)
    target_llm_writing = []
    for i in tqdm(list(range(len(target_df)))):
        assert len(target_df.iloc[i][llm_cols[i % 4]]) > 0
        target_llm_writing.append(target_df.iloc[i][llm_cols[i % 4]])
    target_df['ai_abstract'] = target_llm_writing
    target_df = target_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    assert split in ["train", "val"]

    if split == "train":
        # Source: labeled pos rows 0-1500 (ai), labeled neg rows 2000-8000 (human)
        # Rows 1500-2000 are held out as val calibration positives
        source_positive_texts = source_df.iloc[:1500]['ai_abstract'].dropna().tolist()
        source_negative_texts = source_df.iloc[2000:8000]['human_abstract'].dropna().tolist()

        # Target unlabeled: rows 1500-6000, first alpha% ai, remaining (1-alpha)% human
        u_texts = target_df.iloc[1500:6000]
        u_positive_texts = u_texts.iloc[:int(alpha * len(u_texts))]['ai_abstract'].dropna().tolist()
        u_negative_texts = u_texts.iloc[int(alpha * len(u_texts)):]['human_abstract'].dropna().tolist()

    elif split == "val":
        # Source: calibration positives only (held-out rows 1500-2000 ai); no labeled negatives
        source_positive_texts = source_df.iloc[1500:2000]['ai_abstract'].dropna().tolist()
        source_negative_texts = []

        # Target unlabeled: rows 6500-8000, first alpha% ai, remaining (1-alpha)% human
        val_subset = target_df.iloc[6000:8000]
        u_texts = val_subset.iloc[500:]  # rows 6500-8000
        u_positive_texts = u_texts.iloc[:int(alpha * len(u_texts))]['ai_abstract'].dropna().tolist()
        u_negative_texts = u_texts.iloc[int(alpha * len(u_texts)):]['human_abstract'].dropna().tolist()

    if sentence:
        source_positive_texts, _ = split_into_sentences(source_positive_texts, [0] * len(source_positive_texts))
        if source_negative_texts:
            source_negative_texts, _ = split_into_sentences(source_negative_texts, [0] * len(source_negative_texts))
        u_positive_texts, _ = split_into_sentences(u_positive_texts, [0] * len(u_positive_texts))
        u_negative_texts, _ = split_into_sentences(u_negative_texts, [0] * len(u_negative_texts))

        T_pos = len(u_positive_texts) / alpha if alpha > 0 else float('inf')
        T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else float('inf')
        T = int(min(T_pos, T_neg))
        n_pos = int(alpha * T)
        n_neg = T - n_pos
        u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
        u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))

    print(f"PNU split2: {len(source_negative_texts)} labeled neg | "
          f"alpha = {len(u_positive_texts)} / {len(u_positive_texts) + len(u_negative_texts)}")

    texts  = source_positive_texts + source_negative_texts + u_positive_texts + u_negative_texts
    labels = ([1] * len(source_positive_texts)
              + [2] * len(source_negative_texts)
              + [0] * (len(u_positive_texts) + len(u_negative_texts)))

    assert len(texts) == len(labels)
    return texts, labels


def read_arxiv_split2_PNU_pos_test(split_dir, alpha, split, sentence, inject, seed, n_labeled=0):
    """
    PNU version of read_arxiv_split2.
    Source domain: 2010 writing (fixed path); labeled positives (ai) and negatives (human).
    Target domain: test-year writing (split_dir); treated as fully unlabeled.
    split=='train': source labeled pos (rows 0-1500 ai) + labeled neg (rows 2000-8000 human)
                    + target unlabeled (rows 1500-6000 split by alpha).
    split=='val':   source labeled pos only (rows 1500-2000 ai)
                    + target unlabeled (rows 6500-8000 split by alpha).
    """
    assert seed is not None
    inject = False

    SOURCE_2010_PATH = '/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2010_ai_cs._10000_fronthalf_120b_qwen.parquet'
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen"]

    # Load and prepare source (2010) data
    source_df = pd.read_parquet(SOURCE_2010_PATH)
    source_llm_writing = []
    for i in tqdm(list(range(len(source_df)))):
        assert len(source_df.iloc[i][llm_cols[i % 4]]) > 0
        source_llm_writing.append(source_df.iloc[i][llm_cols[i % 4]])
    source_df['ai_abstract'] = source_llm_writing
    source_df = source_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Load and prepare target (test year) data
    target_df = pd.read_parquet(split_dir)
    target_llm_writing = []
    for i in tqdm(list(range(len(target_df)))):
        assert len(target_df.iloc[i][llm_cols[i % 4]]) > 0
        target_llm_writing.append(target_df.iloc[i][llm_cols[i % 4]])
    target_df['ai_abstract'] = target_llm_writing
    target_df = target_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    assert split in ["train", "val"]

    if split == "train":
        # Source: labeled pos rows 0-1500 (ai), labeled neg rows 2000-8000 (human)
        # Rows 1500-2000 are held out as val calibration positives
        source_positive_texts = target_df.iloc[:1500]['ai_abstract'].dropna().tolist()
        source_negative_texts = source_df.iloc[2000:8000]['human_abstract'].dropna().tolist()

        # Target unlabeled: rows 1500-6000, first alpha% ai, remaining (1-alpha)% human
        u_texts = target_df.iloc[1500:6000]
        u_positive_texts = u_texts.iloc[:int(alpha * len(u_texts))]['ai_abstract'].dropna().tolist()
        u_negative_texts = u_texts.iloc[int(alpha * len(u_texts)):]['human_abstract'].dropna().tolist()

    elif split == "val":
        # Source: calibration positives only (held-out rows 1500-2000 ai); no labeled negatives
        source_positive_texts = target_df.iloc[1500:2000]['ai_abstract'].dropna().tolist()
        source_negative_texts = []

        # Target unlabeled: rows 6500-8000, first alpha% ai, remaining (1-alpha)% human
        val_subset = target_df.iloc[6000:8000]
        u_texts = val_subset.iloc[500:]  # rows 6500-8000
        u_positive_texts = u_texts.iloc[:int(alpha * len(u_texts))]['ai_abstract'].dropna().tolist()
        u_negative_texts = u_texts.iloc[int(alpha * len(u_texts)):]['human_abstract'].dropna().tolist()

    if sentence:
        source_positive_texts, _ = split_into_sentences(source_positive_texts, [0] * len(source_positive_texts))
        if source_negative_texts:
            source_negative_texts, _ = split_into_sentences(source_negative_texts, [0] * len(source_negative_texts))
        u_positive_texts, _ = split_into_sentences(u_positive_texts, [0] * len(u_positive_texts))
        u_negative_texts, _ = split_into_sentences(u_negative_texts, [0] * len(u_negative_texts))

        T_pos = len(u_positive_texts) / alpha if alpha > 0 else float('inf')
        T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else float('inf')
        T = int(min(T_pos, T_neg))
        n_pos = int(alpha * T)
        n_neg = T - n_pos
        u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
        u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))

    print(f"PNU split (source positive = test year): {len(source_negative_texts)} labeled neg | "
          f"alpha = {len(u_positive_texts)} / {len(u_positive_texts) + len(u_negative_texts)}")

    texts  = source_positive_texts + source_negative_texts + u_positive_texts + u_negative_texts
    labels = ([1] * len(source_positive_texts)
              + [2] * len(source_negative_texts)
              + [0] * (len(u_positive_texts) + len(u_negative_texts)))

    assert len(texts) == len(labels)
    return texts, labels

def read_arxiv_split2_PNU(split_dir, alpha, split, sentence, inject, seed, n_labeled=0):
    """
    Like read_arxiv_split2_PNU_pos_test but uses double_rewrite (_v2) folder so that
    alpha% of the known source positives are double mirrors (LLM rewrote the LLM rewrite)
    instead of single mirrors. The _v2 file has double mirrors for the first 2000 rows at
    llm_cols[(i+1)%4]; rows 2000+ only have single mirrors.
    Source domain: 2010 writing (fixed path); labeled positives (ai) and negatives (human).
    Target domain: test-year writing (split_dir); treated as fully unlabeled.
    split=='train': source labeled pos (rows 0-1500, alpha% double-mirror ai) + labeled neg
                    (rows 2000-8000 human) + target unlabeled (rows 1500-6000 split by alpha).
    split=='val':   source labeled pos only (rows 1500-2000, alpha% double-mirror ai)
                    + target unlabeled (rows 6500-8000 split by alpha).
    """
    assert seed is not None
    inject = False

    SOURCE_2010_PATH = '/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2010_ai_cs._10000_fronthalf_120b_qwen.parquet'
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen"]

    # Switch to double_rewrite _v2 file (double mirrors at llm_cols[(i+1)%4] for rows 0-1999)
    # split_dir = split_dir.replace("data_raw", "double_rewrite")
    # split_dir = split_dir.replace("_fronthalf_120b_qwen.parquet", "_0.2_fronthalf_120b_qwen_v2.parquet")

    pct_inject = 0.2  # double mirrors exist for the first 20% of rows (2000 out of 10000)

    # Load and prepare source (2010) data -- only human_abstract is used (for labeled negatives)
    source_df = pd.read_parquet(SOURCE_2010_PATH)
    source_llm_writing = []
    for i in tqdm(list(range(len(source_df)))):
        assert len(source_df.iloc[i][llm_cols[i % 4]]) > 0
        source_llm_writing.append(source_df.iloc[i][llm_cols[i % 4]])
    source_df['ai_abstract'] = source_llm_writing
    source_df = source_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Load target (test year) data and build both single and double mirror columns
    # before shuffling so row-index-based column cycling (i%4, (i+1)%4) is correct.
    target_df = pd.read_parquet(split_dir)
    num_inject = int(pct_inject * len(target_df))  # 2000 rows have double mirrors
    target_llm_writing = []
    target_double_writing = []
    for i in tqdm(list(range(len(target_df)))):
        assert len(target_df.iloc[i][llm_cols[i % 4]]) > 0
        single = target_df.iloc[i][llm_cols[i % 4]]
        target_llm_writing.append(single)
        if i < num_inject:
            double = target_df.iloc[i][llm_cols[(i + 1) % 4]]
            target_double_writing.append(double if isinstance(double, str) and len(double) > 0 else None)
        else:
            target_double_writing.append(None)
    target_df['ai_abstract'] = target_llm_writing
    target_df['double_abstract'] = target_double_writing
    target_df = target_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    assert split in ["train", "val"]

    def _build_positives_with_doubles(pos_df, alpha):
        """Return positive texts using alpha% double mirrors (up to however many are available)."""
        n_double_want = int(alpha * len(pos_df))
        has_double = pos_df['double_abstract'].notna()
        double_rows = pos_df[has_double].reset_index(drop=True)
        single_rows = pos_df[~has_double].reset_index(drop=True)
        n_use_double = min(n_double_want, len(double_rows))
        return (
            double_rows.iloc[:n_use_double]['double_abstract'].tolist() +
            double_rows.iloc[n_use_double:]['ai_abstract'].tolist() +
            single_rows['ai_abstract'].tolist()
        )

    if split == "train":
        # Source: labeled pos rows 0-1500, alpha% are double mirrors; neg rows 2000-8000 (human)
        source_positive_texts = _build_positives_with_doubles(target_df.iloc[:1500].reset_index(drop=True), alpha)
        source_negative_texts = source_df.iloc[2000:8000]['human_abstract'].dropna().tolist()

        # Target unlabeled: rows 1500-6000, first alpha% ai, remaining (1-alpha)% human
        u_texts = target_df.iloc[1500:6000]
        u_positive_texts = u_texts.iloc[:int(alpha * len(u_texts))]['ai_abstract'].dropna().tolist()
        u_negative_texts = u_texts.iloc[int(alpha * len(u_texts)):]['human_abstract'].dropna().tolist()

    elif split == "val":
        # Source: calibration positives from rows 6000-6499 (disjoint from train unlabeled 1500-5999)
        source_positive_texts = _build_positives_with_doubles(target_df.iloc[6000:6500].reset_index(drop=True), alpha)
        source_negative_texts = []

        # Target unlabeled: rows 6500-8000, first alpha% ai, remaining (1-alpha)% human
        val_subset = target_df.iloc[6000:8000]
        u_texts = val_subset.iloc[500:]  # rows 6500-8000
        u_positive_texts = u_texts.iloc[:int(alpha * len(u_texts))]['ai_abstract'].dropna().tolist()
        u_negative_texts = u_texts.iloc[int(alpha * len(u_texts)):]['human_abstract'].dropna().tolist()

    if sentence:
        source_positive_texts, _ = split_into_sentences(source_positive_texts, [0] * len(source_positive_texts))
        if source_negative_texts:
            source_negative_texts, _ = split_into_sentences(source_negative_texts, [0] * len(source_negative_texts))
        u_positive_texts, _ = split_into_sentences(u_positive_texts, [0] * len(u_positive_texts))
        u_negative_texts, _ = split_into_sentences(u_negative_texts, [0] * len(u_negative_texts))

        T_pos = len(u_positive_texts) / alpha if alpha > 0 else float('inf')
        T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else float('inf')
        T = int(min(T_pos, T_neg))
        n_pos = int(alpha * T)
        n_neg = T - n_pos
        u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
        u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))

    print(f"PNU split (pos=test year, double_rewrite): {len(source_negative_texts)} labeled neg | "
          f"alpha = {len(u_positive_texts)} / {len(u_positive_texts) + len(u_negative_texts)}")

    texts  = source_positive_texts + source_negative_texts + u_positive_texts + u_negative_texts
    labels = ([1] * len(source_positive_texts)
              + [2] * len(source_negative_texts)
              + [0] * (len(u_positive_texts) + len(u_negative_texts)))

    assert len(texts) == len(labels)
    return texts, labels

def read_arxiv_split_llm_PNU(split_dir, llm, split, sentence, alpha, gemini, flip, seed, n_labeled=0):
    """
    PNU version of read_arxiv_split_llm.
    `llm` is '|'-separated: first token is the source LLM, second is the target (test) LLM.
    Source domain: alpha=0, PN, llm_source, train-equivalent subset (first 75% of rows).
      Labeled positives: rows 75%-90% within source subset (holding 90%-100% for val).
      Labeled negatives (train only): rows 0-75% within source subset.
    Target domain: alpha, PU, llm_test, split-equivalent unlabeled pool.
    split=='val': only calibration positives (rows 90%-end of source subset) + target unlabeled.
    """
    assert seed is not None

    llm_parts = llm.split('|')
    llm_source = llm_parts[0]
    llm_test   = llm_parts[1] if len(llm_parts) > 1 else llm_parts[0]

    llm_cols = (
        ["Llama 3.3 70b Instruct", "GPT OSS 120b", "Qwen", "Gemini 3 Preview"]
        if not gemini
        else ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"]
    )
    assert llm_source in llm_cols or llm_source == "all", f"{llm_source} not valid"
    assert llm_test   in llm_cols or llm_test   == "all", f"{llm_test} not valid"

    arxiv_data = pd.read_parquet(split_dir)

    # --- Source subset: same as alpha=0, PN, llm_source, split='train' in read_arxiv_split_llm ---
    if llm_source == "all":
        source_subset = None
        for i, llm2 in enumerate(llm_cols):
            tmp = arxiv_data[arxiv_data[llm2].notna() & (arxiv_data[llm2] != "")].reset_index(drop=True)
            assert len(tmp) == 2500
            tmp = tmp.sample(frac=1, random_state=seed).reset_index(drop=True)
            print(len(tmp))
            tmp["llm_writing"] = tmp[llm2]
            tmp = tmp.iloc[:int(len(tmp) * .75)].sample(frac=1/len(llm_cols), random_state=seed).reset_index(drop=True)
            source_subset = tmp if source_subset is None else pd.concat([source_subset, tmp]).reset_index(drop=True)
    else:
        source_subset = arxiv_data[arxiv_data[llm_source].notna() & (arxiv_data[llm_source] != "")].reset_index(drop=True)
        source_subset = source_subset.sample(frac=1, random_state=seed).reset_index(drop=True)
        source_subset = source_subset.iloc[:int(len(source_subset) * .75)]

    source_llm_col   = llm_source if llm_source != "all" else "llm_writing"
    source_llm_texts   = source_subset[source_llm_col].tolist()
    source_human_texts = source_subset['human_abstract'].tolist()
    assert len(source_llm_texts) == len(source_human_texts)

    # --- Target subset: same row slice as alpha, PU, llm_test, split in read_arxiv_split_llm ---
    if llm_test == "all":
        target_subset = None
        for i, llm2 in enumerate(llm_cols):
            tmp = arxiv_data[arxiv_data[llm2].notna() & (arxiv_data[llm2] != "")].reset_index(drop=True)
            assert len(tmp) == 2500
            tmp = tmp.sample(frac=1, random_state=seed).reset_index(drop=True)
            print(len(tmp))
            tmp["llm_writing"] = tmp[llm2]
            if split == "train":
                tmp = tmp.iloc[:int(len(tmp) * .5)].sample(frac=1/len(llm_cols), random_state=seed).reset_index(drop=True)
            elif split == "val":
                tmp = tmp.iloc[int(len(tmp) * .5):int(len(tmp) * .75)].sample(frac=1/len(llm_cols), random_state=seed).reset_index(drop=True)
            target_subset = tmp if target_subset is None else pd.concat([target_subset, tmp]).reset_index(drop=True)
    else:
        target_subset = arxiv_data[arxiv_data[llm_test].notna() & (arxiv_data[llm_test] != "")].reset_index(drop=True)
        target_subset = target_subset.sample(frac=1, random_state=seed).reset_index(drop=True)
        if split == "train":
            target_subset = target_subset.iloc[:int(len(target_subset) * .5)]
        elif split == "val":
            target_subset = target_subset.iloc[int(len(target_subset) * .5):int(len(target_subset) * .75)]

    target_llm_col   = llm_test if llm_test != "all" else "llm_writing"
    target_llm_texts   = target_subset[target_llm_col].tolist()
    target_human_texts = target_subset['human_abstract'].tolist()
    assert len(target_llm_texts) == len(target_human_texts)

    # --- Source positive and labeled-negative texts (depending on flip) ---
    if flip:
        n_u_abs = int(len(source_human_texts) * .75)
        if split == "train":
            source_positive_texts = source_human_texts[n_u_abs:int(len(source_human_texts) * .90)]
            source_labeled_neg    = source_llm_texts[:n_u_abs]
        else:  # val: only calibration positives, no labeled negatives
            source_positive_texts = source_human_texts[int(len(source_human_texts) * .90):]
            source_labeled_neg    = []
    else:
        n_u_abs = int(len(source_llm_texts) * .75)
        if split == "train":
            source_positive_texts = source_llm_texts[n_u_abs:int(len(source_llm_texts) * .90)]
            source_labeled_neg    = source_human_texts[:n_u_abs]
        else:  # val: only calibration positives, no labeled negatives
            source_positive_texts = source_llm_texts[int(len(source_llm_texts) * .90):]
            source_labeled_neg    = []

    # --- Target unlabeled pool: u_positive_texts + u_negative_texts from PU(alpha, split) ---
    if flip:
        n_u_target       = int(len(target_human_texts) * .75)
        u_positive_texts = target_human_texts[:int(n_u_target * alpha)]
        u_negative_texts = target_llm_texts[int(n_u_target * alpha):n_u_target]
    else:
        n_u_target       = int(len(target_llm_texts) * .75)
        u_positive_texts = target_llm_texts[:int(n_u_target * alpha)]
        u_negative_texts = target_human_texts[int(n_u_target * alpha):n_u_target]

    if sentence:
        source_positive_texts, _ = split_into_sentences(source_positive_texts, [1] * len(source_positive_texts))
        if source_labeled_neg:
            source_labeled_neg, _ = split_into_sentences(source_labeled_neg, [0] * len(source_labeled_neg))
        u_positive_texts, _ = split_into_sentences(u_positive_texts, [1] * len(u_positive_texts))
        u_negative_texts, _ = split_into_sentences(u_negative_texts, [0] * len(u_negative_texts))

        T_pos = len(u_positive_texts) / alpha if alpha > 0 else float('inf')
        T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else float('inf')
        T = int(min(T_pos, T_neg))
        n_pos = int(alpha * T)
        n_neg = T - n_pos
        u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
        u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))

    print(f"alpha = {len(u_positive_texts)} / {len(u_positive_texts) + len(u_negative_texts)}")
    print(f"PNU llm: {len(source_labeled_neg)} labeled neg")

    final_texts  = source_positive_texts + source_labeled_neg + u_positive_texts + u_negative_texts
    final_labels = ([1] * len(source_positive_texts)
                    + [2] * len(source_labeled_neg)
                    + [0] * (len(u_positive_texts) + len(u_negative_texts)))

    assert len(final_texts) == len(final_labels)
    return final_texts, final_labels


def read_arxiv_split_xy_PNU(split_dir, llm, split, sentence, alpha, gemini, flip, seed, llm_col, n_labeled=0):
    """
    PNU version of read_arxiv_split_xy.
    Source distribution: human_abstract (positive) and rewrite_X (negative).
    Target distribution: human_abstract and ai_abstract from non-overlapping rows.
    Custom row slices avoid overlap since source and target share the same human text.

    split=='train':
      source positives: human_abstract rows 0-1000 (label=1)
      source negatives: rewrite_X rows 1500-2000 (label=2)
      target unlabeled: human_abstract rows 2000-4000 + ai_abstract rows 4000-6000 (label=0)
    split=='val':
      source positives: human_abstract rows 1000-1500 (label=1, held-out from train)
      target unlabeled: human_abstract rows 6000-7000 + ai_abstract rows 7000-8000 (label=0)
    """
    assert seed is not None
    assert sentence, "read_arxiv_split_xy_PNU requires sentence=True"
    assert split in ["train", "val"]

    arxiv_data = pd.read_parquet(split_dir)
    arxiv_data = arxiv_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    # import pdb; pdb.set_trace()

    if split == "train":
        source_positive_texts = arxiv_data.iloc[:1000]['human_abstract'].dropna().tolist()
        source_negative_texts = arxiv_data.iloc[1500:2000]['rewrite_X'].dropna().tolist()
        u_positive_texts      = arxiv_data.iloc[2000:4000]['human_abstract'].dropna().tolist()
        u_negative_texts      = arxiv_data.iloc[4000:6000][f'rewrite_{llm}'].dropna().tolist()
    else:  # val
        source_positive_texts = arxiv_data.iloc[1000:1500]['human_abstract'].dropna().tolist()
        source_negative_texts = []
        u_positive_texts      = arxiv_data.iloc[6000:7000]['human_abstract'].dropna().tolist()
        u_negative_texts      = arxiv_data.iloc[7000:8000][f'rewrite_{llm}'].dropna().tolist()

    source_positive_texts, _ = split_into_sentences(source_positive_texts, [0] * len(source_positive_texts))
    if source_negative_texts:
        source_negative_texts, _ = split_into_sentences(source_negative_texts, [0] * len(source_negative_texts))
    u_positive_texts, _ = split_into_sentences(u_positive_texts, [0] * len(u_positive_texts))
    u_negative_texts, _ = split_into_sentences(u_negative_texts, [0] * len(u_negative_texts))

    T_pos = len(u_positive_texts) / alpha if alpha > 0 else float('inf')
    T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else float('inf')
    T = int(min(T_pos, T_neg))
    n_pos = int(alpha * T)
    n_neg = T - n_pos
    u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
    u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))

    print(f"PNU xy: {len(source_negative_texts)} labeled neg")

    texts  = source_positive_texts + source_negative_texts + u_positive_texts + u_negative_texts
    labels = ([1] * len(source_positive_texts)
              + [2] * len(source_negative_texts)
              + [0] * (len(u_positive_texts) + len(u_negative_texts)))

    assert len(texts) == len(labels)
    # import pdb; pdb.set_trace()
    return texts, labels
