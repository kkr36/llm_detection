from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import pandas as pd
import json
import spacy
from tqdm import tqdm
import re
import random
import csv
nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
nlp.enable_pipe("senter")

import pandas as pd

def find_cells_with_substring(df, x, case=True):
    """
    Returns a list of (row_index, column_name, cell_value)
    for all cells containing substring x.
    """
    # Work on string version (handles NaNs safely)
    df_str = df.astype(str)
    
    # Boolean mask of matches
    mask = df_str.apply(
        lambda col: col.str.contains(x, case=case, na=False)
    )
    
    results = []
    
    # Iterate over True locations
    for row_idx, col_name in zip(*mask.to_numpy().nonzero()):
        results.append(
            (
                df.index[row_idx],
                df.columns[col_name],
                df.iat[row_idx, col_name]
            )
        )
    
    return results

# def clean_text(text):
#     # replace bad things we know how to; remove all other non-typable
#     bad_dashes = ['—', '⁻', '–', '‑', '‐', '−']
#     bad_apostrophes = ['’', '′', '‘']
#     bad_left_quote = "“"
#     bad_right_quote = "”"
#     for bad_dash in bad_dashes:
#         text = [t.replace(bad_dash, '-') for t in text]
#     for bad_apostrophe in bad_apostrophes:
#         text = [t.replace(bad_apostrophe, "'") for t in text]
#     text = [t.replace(bad_left_quote, '"') for t in text]
#     text = [t.replace(bad_right_quote, '"') for t in text]
#     text = [t.replace("\n", " ") for t in text]

#     # remove non-typable
#     allowed = set(string.printable)
#     text = [''.join(ch for ch in s if ch in allowed) for s in text]
#     return text

# import math

# def split_into_sentences(abstracts, labels):
#     all_sentences = []
#     all_labels = []
#     # abstracts = [t.replace('‑', '-').replace('’', "'") for t in abstracts]

#     # all_sentences_2 = []
#     # all_labels_2 = []
#     print("Splitting into sentences!")
#     for i, abstract in tqdm(list(enumerate(abstracts))):
#         doc = nlp(abstract)
#         sentences = [sent.text.strip() for sent in doc.sents]
#         # sentences = [s[i:i+len(s)//2 + len(s)%2] for s in sentences for i in (0, len(s)//2 + len(s)%2)]
#         # sentences = sentences[:2] + sentences[-2:]
#         # sentences = [x for x in sentences if len(x) > 1]
#         all_sentences.extend(sentences)  # extend instead of append
#         all_labels.extend([labels[i] for _ in range(len(sentences))])
#         # all_sentences_2.extend(sentences2)
#         # all_labels_2.extend([labels[i] for _ in range(len(sentences2))])
#     print(f"Made {len(all_sentences)} sentences")
#     # import pdb; pdb.set_trace()
#     return all_sentences, all_labels


def create_blind_test(human_text, llm_text, n, title, seed=42):
    """
    Samples n entries from each list, shuffles them together,
    and writes key.csv and test.csv.
    """
    assert n <= len(human_text), "n larger than human_text size"
    assert n <= len(llm_text), "n larger than llm_text size"

    random.seed(seed)

    # Sample n from each
    human_sample = random.sample(human_text, n)
    llm_sample = random.sample(llm_text, n)

    # import pdb; pdb.set_trace()

    # Label them
    combined = (
        [(text, "human") for text in human_sample] +
        [(text, "LLM") for text in llm_sample]
    )

    # Shuffle combined list
    random.shuffle(combined)

    # Write key.csv
    with open(f"{title}_key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "writer"])
        for text, label in combined:
            writer.writerow([text, label])

    # Write test.csv (same order, no labels)
    with open(f"{title}_test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for text, _ in combined:
            writer.writerow([text])

def split_into_sentences(abstracts, labels, batch_size=200, n_process=1):
    all_sentences = []
    all_labels = []

    print("Splitting into sentences!")

    # Wrap nlp.pipe with tqdm for progress
    for doc, label in zip(tqdm(nlp.pipe(abstracts, batch_size=batch_size, n_process=n_process),
                                total=len(abstracts),
                                desc="Processing abstracts"),
                          labels):
        sentences = [sent.text.strip() for sent in doc.sents]
        all_sentences.extend(sentences)
        all_labels.extend([label] * len(sentences))

    print(f"Made {len(all_sentences)} sentences")
    return all_sentences, all_labels


def read_paramveer(split_dir, split="train", ft=False):
    with open(split_dir, 'r') as f:
        data = json.load(f)

    ai_data, ai_labels = [], []
    human_data, human_labels = [], []
    ft_data, ft_labels = [], []

    for excerpt in data:
        human_writing = excerpt['MFA']
        ai_writing = excerpt['AI']
        for human_sample in human_writing.values():
            doc = nlp(human_sample)
            sentences = [sent.text.strip() for sent in doc.sents]
            human_data.extend(sentences)  # extend instead of append
            human_labels.extend([0 for _ in range(len(sentences))])
        for ai_sample in ai_writing.values():
            doc = nlp(ai_sample)
            sentences = [sent.text.strip() for sent in doc.sents]
            ai_data.extend(sentences)  # extend instead of append
            ai_labels.extend([1 for _ in range(len(sentences))])
        # import pdb; pdb.set_trace()
        if 'GPT4_Finetuned' in excerpt:
            ft_writing = excerpt['GPT4_Finetuned']
            doc = nlp(ft_writing)
            sentences = [sent.text.strip() for sent in doc.sents]
            ft_data.extend(sentences)  # extend instead of append
            ft_labels.extend([1 for _ in range(len(sentences))])
        else:
            print(f"{excerpt['writer']} no ft example")
    
    
    n_ai = len(ai_data)
    n_human = len(human_data)
    n_ft = len(ft_data)

    if split=="train":
        ai_data, ai_labels = ai_data[:int(n_ai * .7)], ai_labels[:int(n_ai * .7)]
        human_data, human_labels = human_data[:int(n_human * .8)], human_labels[:int(n_human * .8)]
        ft_data, ft_labels = ft_data[:int(n_ft * .7)], ft_labels[:int(n_ft * .7)]

    elif split=="val":
        ai_data, ai_labels = ai_data[int(n_ai * .8):], ai_labels[int(n_ai * .8):]
        human_data, human_labels = human_data[int(n_human * .8):], human_labels[int(n_human * .8):]
        ft_data, ft_labels = ft_data[int(n_ft * .8):], ft_labels[int(n_ft * .8):]

    elif split=="test":
        pos_ai_data = ai_data[int(n_ai*.7):int(n_ai*.8)]
        pos_ft_data = ft_data[int(n_ft*.7):int(n_ft*.8)]
        u_ai_data = ai_data[int(n_ai*.8):]
        u_ft_data = ft_data[int(n_ft*.8):]
        val_human_data = human_data[int(n_human*.8):]

        if ft:
            data, labels = {
                "pos_ai": pos_ft_data, # once pos_ft_data
                "u_ai": u_ft_data,
                "u_human": val_human_data
            }, \
            {
                "pos_ai": [1 for _ in range(len(pos_ft_data))],
                "u_ai": [-1 for _ in range(len(u_ft_data))],
                "u_human": [0 for _ in range(len(val_human_data))]
            }
            # return ft_data + pos_ai_data, [0 for _ in range(len(ft_data))] + [1 for _ in range(len(pos_ai_data))]
        else:
            data, labels =  {
                "pos_ai": pos_ai_data,
                "u_ai": u_ai_data,
                "u_human": val_human_data
            }, \
            {
                "pos_ai": [1 for _ in range(len(pos_ai_data))],
                "u_ai": [-1 for _ in range(len(u_ai_data))],
                "u_human": [0 for _ in range(len(val_human_data))]
            }
        # return data["pos_ai"] + data["u_ai"] + data["u_human"], labels["pos_ai"] + labels["u_ai"] + labels["u_human"]
        return data["pos_ai"] + data["u_ai"] + data["u_human"], labels["pos_ai"] + labels["u_ai"] + labels["u_human"]

            # return u_ai_data + pos_ai_data, [0 for _ in range(len(u_ai_data))] + [1 for _ in range(len(pos_ai_data))]
    
    # train/val
    if ft:
        # if split == "train": import pdb; pdb.set_trace()
        return ft_data + human_data, ft_labels + human_labels
    else:
        return ai_data + human_data, ai_labels + human_labels

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir=="neg" else 1)

    return texts, labels

def read_arxiv_split(split_dir):
    df = pd.read_parquet(split_dir)
    ai_data, ai_labels = df['ai_sentence'].tolist(), [1 for _ in range(len(df))]
    human_data = df[df['human_index'] != -1]['human_sentence'].tolist()
    human_labels = [0 for _ in range(len(human_data))]
    ai_data, ai_labels = ai_data[:len(human_labels)], ai_labels[:len(human_labels)] # shorten things so classes are balanced, making math easier later
    texts = human_data + ai_data
    labels = human_labels + ai_labels

    return texts, labels

def remove_first_last_sentence(paragraph):
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    if len(sentences) <= 2:
        return ""  # or paragraph, depending on your preference
    return " ".join(sentences[1:-1])

def read_arxiv_split_year(data_dir, label_year, unlabel_year, alpha=None, split="train", sentence=False):
    data_path_label = f'{data_dir}/multillm/data_raw/arxiv_{label_year}_ai_cs._10000_fronthalf.parquet'
    data_path_unlabel = f'{data_dir}/multillm/data_raw/arxiv_{unlabel_year}_ai_cs._10000_fronthalf.parquet'

    arxiv_data_positive = pd.read_parquet(data_path_label)
    arxiv_data_unlabel = pd.read_parquet(data_path_unlabel)

    if split=="train":
        arxiv_data_positive = arxiv_data_positive.iloc[:int(len(arxiv_data_positive)*.7)]
        arxiv_data_unlabel = arxiv_data_unlabel.iloc[:int(len(arxiv_data_unlabel)*.7)]
        assert(len(arxiv_data_positive) == 7000 and len(arxiv_data_unlabel) == 7000)
    elif split=="val":
        arxiv_data_positive = arxiv_data_positive.iloc[int(len(arxiv_data_positive)*.7):]
        arxiv_data_unlabel = arxiv_data_unlabel.iloc[int(len(arxiv_data_unlabel)*.7):]
        assert(len(arxiv_data_positive) == 3000 and len(arxiv_data_unlabel) == 3000)

    # for the unlabeled data, replace alpha % with llm mirror
    unlabeled_writing = []

    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
    inject_counter = int(alpha * len(arxiv_data_unlabel))
    for i in tqdm(list(i for i in range(len(arxiv_data_unlabel)))):
        assert(len(arxiv_data_unlabel.iloc[i][llm_cols[i % 4]]) > 0)
        mirror = arxiv_data_unlabel.iloc[i][llm_cols[i % 4]]
        if inject_counter:
            unlabeled_writing.append(mirror)
            inject_counter -= 1
        else:
            abstract = arxiv_data_unlabel.iloc[i]['human_abstract']
            unlabeled_writing.append(abstract)
    
    positive_writing = arxiv_data_positive['human_abstract'].tolist()
    texts = positive_writing + unlabeled_writing
    labels = [1 for _ in range(len(positive_writing))] + [0 for _ in range(len(unlabeled_writing))]
    return texts, labels

def read_arxiv_single_double(split_dir, split, sentence, inject, seed):
    assert(seed is not None)
    assert(False)

    pct_inject = .2
    if not inject: pct_inject = 0
    
    arxiv_data = pd.read_parquet(split_dir)

    num_inject = int(pct_inject * len(arxiv_data))
    print(f"injecting {num_inject} LLM-written abstracts ({pct_inject}) into pool of {len(arxiv_data)} abstracts")

    llm_writing = []
    inject_counter = num_inject
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
    for i in tqdm(list(i for i in range(len(arxiv_data)))):
        assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
        original_rewrite = arxiv_data.iloc[i][llm_cols[i % 4]]

        if inject_counter > 0 and arxiv_data.iloc[i][llm_cols[(i+1)%4]] is None:
            import pdb; pdb.set_trace()
        if inject_counter > 0 and len(arxiv_data.iloc[i][llm_cols[(i+1)%4]]) > 0:
            if not inject:
                assert(False), "should not be injecting"
            mirror = arxiv_data.iloc[i][llm_cols[(i+1)%4]]
            llm_writing.append(mirror)
            arxiv_data.at[i, 'human_abstract'] = original_rewrite
            inject_counter -= 1
        else:
            # assert(False)
            llm_writing.append(original_rewrite)
    arxiv_data['ai_abstract'] = llm_writing
    assert(inject_counter == 0)

    wrong_labels = arxiv_data.iloc[:num_inject].reset_index(drop=True)
    wrong_labels = wrong_labels.sample(frac=1, random_state=seed).reset_index(drop=True)

    assert(split in ["train", "val"])
    if split == "train":
        data = wrong_labels.iloc[:int(len(wrong_labels)*.75)]
    elif split == "val":
        data = wrong_labels.iloc[int(len(wrong_labels)*.75):]

    if sentence:
        single_texts, single_labels = split_into_sentences(data['human_abstract'].tolist(), [0 for _ in range(len(data))])
        double_texts, double_labels = split_into_sentences(data['ai_abstract'].tolist(), [1 for _ in range(len(data))])

        texts = single_texts + double_texts
        labels = single_labels + double_labels
    
    else:
        human_texts = data["human_abstract"].dropna().tolist()
        ai_texts = data["ai_abstract"].dropna().tolist()
        texts = human_texts + ai_texts
        labels = [0 for _ in range(len(human_texts))] + [1 for _ in range(len(ai_texts))]

    assert(len(texts) == len(labels))
    return texts, labels

def read_arxiv_split2(split_dir, alpha, split, sentence, inject, seed):
    assert(seed is not None)
    # import os
    # if not(os.path.exists(split_dir)):
    # year = int(re.search(r"\d{4}", split_dir).group(0))
    # pct_inject = 0 if (year < 2014 or inject is False) else 0.05 * (year - 2012) / 2
    pct_inject = .2
    if not inject: pct_inject = 0
    
    arxiv_data = pd.read_parquet(split_dir)

    # import pdb; pdb.set_trace()
    # tmp = arxiv_data.copy(deep=True)
    num_inject = int(pct_inject * len(arxiv_data))
    print(f"injecting {num_inject} LLM-written abstracts ({pct_inject}) into pool of {len(arxiv_data)} abstracts")

    llm_writing = []
    inject_counter = num_inject
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
    for i in tqdm(list(i for i in range(len(arxiv_data)))):
        assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
        original_rewrite = arxiv_data.iloc[i][llm_cols[i % 4]]
        # original_rewrite = "THIS IS AI WRITING -- I'm Watermarking here!!"
        # import pdb; pdb.set_trace()
        # if inject_counter > 0: 
        #     import pdb; pdb.set_trace()
        if inject_counter > 0 and arxiv_data.iloc[i][llm_cols[(i+1)%4]] is None:
            import pdb; pdb.set_trace()
        if inject_counter > 0 and len(arxiv_data.iloc[i][llm_cols[(i+1)%4]]) > 0:
            if not inject:
                assert(False), "should not be injecting"
            mirror = arxiv_data.iloc[i][llm_cols[(i+1)%4]]
            # mirror = "THIS IS AI WRITING -- I'm Watermarking here!!"
            llm_writing.append(mirror)
            arxiv_data.at[i, 'human_abstract'] = original_rewrite
            # arxiv_data.at[i, 'human_abstract'] = np.nan
            inject_counter -= 1
        else:
            llm_writing.append(original_rewrite)
    arxiv_data['ai_abstract'] = llm_writing
    assert(inject_counter == 0)

    wrong_labels = arxiv_data.iloc[:num_inject].reset_index(drop=True)
    right_labels = arxiv_data.iloc[num_inject:].reset_index(drop=True)

    wrong_labels = wrong_labels.sample(frac=1, random_state=seed).reset_index(drop=True)
    right_labels = right_labels.sample(frac=1, random_state=seed).reset_index(drop=True)

    assert(split in ["train", "val"])
    if split == "train":
        x = int(500 * (3 - 5 * alpha))
        y = 5000 - x
        wrong_labels_subset = wrong_labels.iloc[:int(len(wrong_labels)*.75)-x]
        right_labels_subset = right_labels.iloc[:int(len(right_labels)*.75)-y] # was once just -5000
        subset = pd.concat([wrong_labels_subset, right_labels_subset]).reset_index(drop=True)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # subset = arxiv_data.iloc[:int(len(arxiv_data)*.75)]
        # subset = wrong_labels.iloc[:int(len(wrong_labels)*.75)]
    elif split == "val":
        x = int(500 - ((500 / .6) * alpha))
        y = 1666 - x
        wrong_labels_subset = wrong_labels.iloc[int(len(wrong_labels)*.75)+x:]
        right_labels_subset = right_labels.iloc[int(len(right_labels)*.75)+y:] # was once just + 1666
        subset = pd.concat([wrong_labels_subset, right_labels_subset]).reset_index(drop=True)
        # import pdb; pdb.set_trace()
    
        # subset = arxiv_data.iloc[int(len(arxiv_data)*.75):]
    # if num_inject > 0:
        # import pdb; pdb.set_trace()

    # print("removing sentences")
    # texts = [remove_first_last_sentence(abs) for abs in texts]
    # print("removed sentences")
    # if split=="val": import pdb; pdb.set_trace()
    if sentence:

        positive_texts = wrong_labels_subset["ai_abstract"].dropna().tolist() + right_labels_subset["ai_abstract"].dropna().tolist()
        u_positive_texts = wrong_labels_subset["human_abstract"].dropna().tolist()
        u_negative_texts = right_labels_subset["human_abstract"].dropna().tolist()

        positive_texts, _ = split_into_sentences(positive_texts, [0 for _ in range(len(positive_texts))])
        u_positive_texts, _ = split_into_sentences(u_positive_texts, [0 for _ in range(len(u_positive_texts))])
        u_negative_texts, _ = split_into_sentences(u_negative_texts, [0 for _ in range(len(u_negative_texts))])
        # import pdb; pdb.set_trace()
        # create_blind_test(clean_text(u_negative_texts),clean_text(positive_texts),20,"sentence")
        # create_blind_test(clean_text(right_labels['human_abstract'].tolist()),clean_text(right_labels['ai_abstract'].tolist()),10,"abstract")
        # exit()
        # Compute feasible T bounds
        T_pos = len(u_positive_texts) / alpha if alpha > 0 else np.inf
        T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else np.inf

        T = int(min(T_pos, T_neg))

        n_pos = int(alpha * T)
        n_neg = T - n_pos  # ensures n_pos + n_neg = T exactly

        # import pdb; pdb.set_trace()

        u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
        u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))
        # import pdb; pdb.set_trace()

        print(f"alpha = {len(u_positive_texts)} / {len(u_positive_texts) + len(u_negative_texts)} = {len(u_positive_texts) / (len(u_positive_texts) + len(u_negative_texts))}")


        texts = positive_texts + u_positive_texts + u_negative_texts
        labels = [1 for _ in range(len(positive_texts))] + [0 for _ in range(len(u_positive_texts) + len(u_negative_texts))]
    
    else:
        human_texts = subset["human_abstract"].dropna().tolist()
        ai_texts = subset["ai_abstract"].dropna().tolist()
        texts = human_texts + ai_texts
        labels = [0 for _ in range(len(human_texts))] + [1 for _ in range(len(ai_texts))]
        print(f"Pollution {split}: {len(wrong_labels_subset)} / {len(wrong_labels_subset)}+ {len(right_labels_subset)} = {len(wrong_labels_subset) / (len(right_labels_subset) + len(wrong_labels_subset))}")

    assert(len(texts) == len(labels))
    # np.random.seed(42)
    # perm = np.random.permutation(len(texts))
    # texts = np.array(texts)[perm].tolist()
    # labels = np.array(labels)[perm].tolist()
    # import pdb; pdb.set_trace()
    return texts, labels

def read_arxiv_split_add(split_dir, alpha, split, sentence, inject, seed):
    assert(inject)
    assert(seed is not None)
    pct_inject = .2
    
    arxiv_data = pd.read_parquet(split_dir)
    num_inject = int(pct_inject * len(arxiv_data))
    print(f"injecting {num_inject} LLM-written abstracts ({pct_inject}) into pool of {len(arxiv_data)} abstracts")

    llm_writing = []
    inject_counter = num_inject
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
    for i in tqdm(list(i for i in range(len(arxiv_data)))):
        assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
        original_rewrite = arxiv_data.iloc[i][llm_cols[i % 4]]

        if inject_counter > 0 and arxiv_data.iloc[i][llm_cols[(i+1)%4]] is None:
            import pdb; pdb.set_trace()
        if inject_counter > 0 and len(arxiv_data.iloc[i][llm_cols[(i+1)%4]]) > 0:
            if not inject:
                assert(False), "should not be injecting"
            mirror = arxiv_data.iloc[i][llm_cols[(i+1)%4]]
            llm_writing.append(mirror)
            arxiv_data.at[i, 'human_abstract'] = original_rewrite
            inject_counter -= 1
        else:
            llm_writing.append(original_rewrite)
    arxiv_data['ai_abstract'] = llm_writing
    assert(inject_counter == 0)


    wrong_labels = arxiv_data.iloc[:num_inject].reset_index(drop=True)
    right_labels = arxiv_data.iloc[num_inject:int(num_inject + (2/3*num_inject))].reset_index(drop=True)

    wrong_labels = wrong_labels.sample(frac=1, random_state=seed).reset_index(drop=True)
    right_labels = right_labels.sample(frac=1, random_state=seed).reset_index(drop=True)

    assert(split in ["train", "val"])
    if split == "train":
        wrong_labels_subset = wrong_labels.iloc[:int(len(wrong_labels)*.75)].reset_index(drop=True)
        right_labels_subset = right_labels.iloc[:int(len(right_labels)*.75)].reset_index(drop=True)

    elif split == "val":

        wrong_labels_subset = wrong_labels.iloc[int(len(wrong_labels)*.75):].reset_index(drop=True)
        right_labels_subset = right_labels.iloc[int(len(right_labels)*.75):].reset_index(drop=True)
    
    num_bad_labels = int((alpha * len(right_labels_subset)) / (1 - alpha))
    wrong_labels_subset = wrong_labels_subset.iloc[:num_bad_labels]

    subset = pd.concat([wrong_labels_subset, right_labels_subset]).reset_index(drop=True)

    print(f"pollution pre sentence: {num_bad_labels / (num_bad_labels + len(right_labels_subset))}")

    if sentence:
        wrong_texts_human = wrong_labels_subset["human_abstract"].dropna().tolist()
        wrong_texts_ai = wrong_labels_subset["ai_abstract"].dropna().tolist()
        right_texts_human, right_texts_ai = right_labels_subset['human_abstract'].dropna().tolist(), right_labels_subset['ai_abstract'].dropna().tolist()

        human_texts, human_labels = wrong_texts_human + right_texts_human, [1 for _ in range(len(wrong_texts_human))] + [0 for _ in range(len(right_texts_human))]
        human_texts, human_labels = split_into_sentences(human_texts, human_labels)

        # if alpha not in [0, 1]:
        #     human_labels = np.array(human_labels)
        #     rng = np.random.default_rng(42)
        #     n_unlabeled = min(len(np.where(human_labels==1)[0]) / alpha, len(np.where(human_labels==0)[0]) / (1-alpha))
        #     idx = np.concatenate([rng.choice(np.where(human_labels==k)[0], int(n_unlabeled*(alpha if k==0 else 1-alpha)), False) for k in (0,1)])
        #     human_texts, human_labels = np.array(human_texts)[idx].tolist(), np.array(human_labels)[idx].tolist()

        print(f"Pollution add {split}: {sum(human_labels)} / {len(human_labels)} = {sum(human_labels) / len(human_labels)}")

        ai_texts, ai_labels = wrong_texts_ai + right_texts_ai, [1 for _ in range(len(wrong_texts_ai) + len(right_texts_ai))]
        ai_texts, ai_labels = split_into_sentences(ai_texts, ai_labels)

        texts, labels = human_texts + ai_texts, [0 for _ in range(len(human_texts))] + [1 for _ in range(len(ai_texts))]

    else:
        human_texts = subset["human_abstract"].dropna().tolist()
        ai_texts = subset["ai_abstract"].dropna().tolist()
        texts = human_texts + ai_texts
        labels = [0 for _ in range(len(human_texts))] + [1 for _ in range(len(ai_texts))]
        print(f"Pollution {split}: {len(wrong_labels_subset)} / {len(wrong_labels_subset)}+ {len(right_labels_subset)} = {len(wrong_labels_subset) / (len(right_labels_subset) + len(wrong_labels_subset))}")

    assert(len(texts) == len(labels))

    return texts, labels

def read_arxiv_split_llm(split_dir, llm, split, sentence, alpha, gemini, flip, seed):
    assert(seed is not None)
    # import pdb; pdb.set_trace()
    llm_cols = ["Llama 3.3 70b Instruct", "GPT OSS 120b", "Gemini 2.5 Flash", "Gemini 3 Preview"] if not gemini else ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"]
    assert(llm in llm_cols or llm=="all"), f"{llm} not valid"

    # flip = False
    
    arxiv_data = pd.read_parquet(split_dir)

    if llm=="all":
        llm_subset=None

        for i, llm2 in enumerate(llm_cols):
            tmp_subset = arxiv_data[arxiv_data[llm2].notna() & (arxiv_data[llm2] != "")].reset_index(drop=True)
            assert(len(tmp_subset)==2500)

            llm_subset = llm_subset.sample(frac=1, random_state=seed).reset_index(drop=True)

            # tmp_subset = tmp_subset.iloc[:int(2500*.75)] #3k total; if all 4 llms then 750
            print(len(tmp_subset))
            tmp_subset["llm_writing"] = tmp_subset[llm2]

            if split=="train":
                tmp_subset = tmp_subset.iloc[:int(len(tmp_subset)*.75)]
            elif split=="val":
                tmp_subset = tmp_subset.iloc[int(len(tmp_subset)*.75):]

            if llm_subset is None:
                llm_subset = tmp_subset
            else:
                llm_subset = pd.concat([llm_subset, tmp_subset]).reset_index(drop=True)
    else:
        llm_subset = arxiv_data[arxiv_data[llm].notna() & (arxiv_data[llm] != "")].reset_index(drop=True) # isolate llm writing

        # shuffle
        llm_subset = llm_subset.sample(frac=1, random_state=seed).reset_index(drop=True)

        if split=="train":
            llm_subset = llm_subset.iloc[:int(len(llm_subset)*.75)]
        elif split=="val":
            llm_subset = llm_subset.iloc[int(len(llm_subset)*.75):]

    llm_texts = llm_subset[llm if llm != "all" else "llm_writing"].tolist()
    human_texts = llm_subset['human_abstract'].tolist()
    assert(len(llm_texts) == len(human_texts))

    if flip:
        # import pdb; pdb.set_trace()
        positive_texts = human_texts[int(len(human_texts)*.75):]
        u_positive_texts = human_texts[:int(len(human_texts)*.75)]
        u_negative_texts = llm_texts
    else:
        positive_texts = llm_texts[int(len(llm_texts)*.75):]
        u_positive_texts = llm_texts[:int(len(llm_texts)*.75)]
        u_negative_texts = human_texts

    # sentence check
    if sentence:
        
        positive_texts, positive_labels = split_into_sentences(positive_texts, [1 for _ in range(len(positive_texts))])
        u_positive_texts, u_positive_labels = split_into_sentences(u_positive_texts, [1 for _ in range(len(u_positive_texts))])
        u_negative_texts, u_negative_labels = split_into_sentences(u_negative_texts, [0 for _ in range(len(u_negative_texts))])

    # min_size = min(len(u_positive_texts), len(u_negative_texts))
    # n_pos = int(min_size * alpha)
    # n_neg = int(min_size * (1-alpha))

    # Compute feasible T bounds
    T_pos = len(u_positive_texts) / alpha if alpha > 0 else np.inf
    T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else np.inf

    T = int(min(T_pos, T_neg))

    n_pos = int(alpha * T)
    n_neg = T - n_pos  # ensures n_pos + n_neg = T exactly

    u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
    u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))
    print(f"alpha = {len(u_positive_texts)} / {len(u_positive_texts) + len(u_negative_texts)} = {len(u_positive_texts) / (len(u_positive_texts) + len(u_negative_texts))}")


    final_texts = positive_texts + u_positive_texts + u_negative_texts
    final_labels = [1 for _ in range(len(positive_texts))] + [0 for _ in range(len(u_positive_texts) + len(u_negative_texts))]

    assert(len(final_texts) == len(final_labels))
    return final_texts, final_labels

def read_arxiv_split_xy(split_dir, llm, split, sentence, alpha, gemini, flip, seed):
    assert(seed is not None)
    assert("pu" in split or "pn" in split or split == "cal")

    # load in data
    arxiv_data = pd.read_parquet(split_dir)
    arxiv_data = arxiv_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    if "pu" in split:
        assert(flip)
        human_writing = arxiv_data.iloc[:4000]["human_abstract"].tolist()
        u_positive_texts = arxiv_data.iloc[4000:4000+int(4000*alpha)]["human_abstract"].tolist()
        u_negative_texts = arxiv_data.iloc[4000+int(4000*alpha):8000]["rewrite_Y"].tolist()

        if "train" in split:
            human_writing = human_writing[:3000]
            u_positive_texts = u_positive_texts[:-int(1000*alpha)]
            u_negative_texts = u_negative_texts[:-int(1000*(1-alpha))]
        elif "val" in split:
            human_writing = human_writing[3000:]
            u_positive_texts = u_positive_texts[-int(1000*alpha):]
            u_negative_texts = u_negative_texts[-int(1000*(1-alpha)):]

    elif "pn" in split:
        assert(alpha == 0 and not flip)
        human_writing = arxiv_data.iloc[:4000]["human_abstract"].tolist()
        ai_writing = arxiv_data.iloc[4000:8000]["rewrite_X"].tolist()

    elif "cal" in split:
        human_writing = arxiv_data.iloc[-2000:]["human_abstract"].tolist()
        ai_writing = arxiv_data.iloc[-2000:]["rewrite_Y"].tolist()


    if sentence:
        if "pn" in split or "cal" in split:
            negative_texts, _ = split_into_sentences(human_writing, [0 for _ in range(len(human_writing))])
            positive_texts, _ = split_into_sentences(ai_writing, [0 for _ in range(len(ai_writing))])

            texts = negative_texts + positive_texts
            labels = [0 for _ in range(len(negative_texts))] + [1 for _ in range(len(positive_texts))]

        elif "pu" in split:

            positive_texts, _ = split_into_sentences(human_writing, [0 for _ in range(len(human_writing))])
            u_positive_texts, _ = split_into_sentences(u_positive_texts, [0 for _ in range(len(u_positive_texts))])
            u_negative_texts, _ = split_into_sentences(u_negative_texts, [0 for _ in range(len(u_negative_texts))])

            # Compute feasible T bounds
            T_pos = len(u_positive_texts) / alpha if alpha > 0 else np.inf
            T_neg = len(u_negative_texts) / (1 - alpha) if alpha < 1 else np.inf

            T = int(min(T_pos, T_neg))

            n_pos = int(alpha * T)
            n_neg = T - n_pos  # ensures n_pos + n_neg = T exactly

            u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
            u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))

            print(f"alpha = {len(u_positive_texts)} / {len(u_positive_texts) + len(u_negative_texts)} = {len(u_positive_texts) / (len(u_positive_texts) + len(u_negative_texts))}")

            texts = positive_texts + u_positive_texts + u_negative_texts
            labels = [1 for _ in range(len(positive_texts))] + [0 for _ in range(len(u_positive_texts) + len(u_negative_texts))]
        
    else:
        assert(False)
        # if "pn" in split or "cal" in split:
        #     texts = human_writing + ai_writing
        #     labels = [0 for _ in range(len(human_writing))] + [1 for _ in range(len(ai_writing))]
        #     print(f"Pollution {split}: {len(wrong_labels_subset)} / {len(wrong_labels_subset)}+ {len(right_labels_subset)} = {len(wrong_labels_subset) / (len(right_labels_subset) + len(wrong_labels_subset))}")
        # elif "pu" in split:

    # import pdb; pdb.set_trace()


    assert(len(texts) == len(labels))
    return texts, labels


def arxiv_len_eda(sentence=False):
    year = 2020
    data_dir = '/share/garg/arxiv_kaggle'
    split_dir = f'{data_dir}/multillm/double_rewrite/arxiv_{year}_ai_cs._10000_0.2_fronthalf.parquet'

    from matplotlib import pyplot as plt
    
    arxiv_data = pd.read_parquet(split_dir)

    llm_writing = []
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
    for i in tqdm(list(i for i in range(len(arxiv_data)))):
        assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
        original_rewrite = arxiv_data.iloc[i][llm_cols[i % 4]]
        llm_writing.append(original_rewrite)
    arxiv_data['ai_abstract'] = llm_writing

    if sentence:
        print("splitting into sentences")
        texts_human, _ = split_into_sentences(arxiv_data["human_abstract"].tolist(), [0 for _ in range(len(arxiv_data))])
        texts_ai, _ = split_into_sentences(arxiv_data["ai_abstract"].tolist(), [1 for _ in range(len(arxiv_data))])
    else:
        texts_human = arxiv_data["human_abstract"].tolist()
        texts_ai = arxiv_data["ai_abstract"].tolist()

    human_lens, ai_lens = np.array([len(x) for x in texts_human]), np.array([len(x) for x in texts_ai])

    bins = np.linspace(
        min(human_lens.min(), ai_lens.min()),
        max(human_lens.max(), ai_lens.max()),
        50
    )

    plt.figure()
    plt.hist(ai_lens, bins=bins, alpha=0.3, density=True, label="Positive")
    plt.hist(human_lens, bins=bins, alpha=0.3, density=True, label="Negative")
    plt.legend()
    plt.savefig(f"length_eda_{'sentence' if sentence else 'abstract'}.pdf", format='pdf')
    plt.clf()


def arxiv_len_eda_llm(sentence=False):
    year = 2020
    data_dir = '/share/garg/arxiv_kaggle'
    split_dir = f'{data_dir}/multillm/double_rewrite/arxiv_{year}_ai_cs._10000_0.2_fronthalf.parquet'

    from matplotlib import pyplot as plt
    
    arxiv_data = pd.read_parquet(split_dir)

    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
    length_dict = {}
    for llm in llm_cols:
        llm_subset = arxiv_data[arxiv_data[llm].notna() & (arxiv_data[llm] != "")].reset_index(drop=True)
        if sentence:
            llm_texts, _ = split_into_sentences(llm_subset[llm].tolist(), [1 for _ in range(len(llm_subset))])
            length_dict[llm] = [len(x) for x in llm_texts]
        else:
            llm_lengths = [len(x) for x in llm_subset[llm].tolist()]
            length_dict[llm] = llm_lengths
    
    global_min = min(min(x) for x in length_dict.values())
    global_max = max(max(x) for x in length_dict.values())
    print(global_min, global_max)
    bins = np.linspace(
        global_min,
        global_max,
        75
    )
    plt.figure()
    for llm in llm_cols:
        plt.hist(length_dict[llm], bins=bins, alpha=0.3, density=True, label=llm)
    plt.legend()
    plt.savefig(f"llms_length_eda_{'sentence' if sentence else 'abstract'}.pdf", format='pdf')
    plt.clf()

def getBertTokenizer(model):
    if model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f'Model: {model} not recognized.')

    return tokenizer

def initialize_bert_transform(net):
    # assert 'bert' in config.model
    # assert config.max_token_length is not None

    tokenizer = getBertTokenizer(net)
    def transform(text):
        tokens = tokenizer(
            text,
            padding=True,
            truncation=True)
        if net == 'bert-base-uncased':
            x = np.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                axis=2)
        elif net == 'distilbert-base-uncased':
            x = np.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                axis=2)
        # x = np.squeeze(x) # First shape dim is always 1
        return x
    return transform

class IMDbBERTData(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        labels = np.array(labels)
        
        encodings = transform(data)

        p_data_idx = np.where(labels==1)[0]
        n_data_idx = np.where(labels==0)[0]
        
        self.p_data = encodings[p_data_idx, :, :]
        self.n_data = encodings[n_data_idx, :, :]

        self.labels = labels

        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.labels)

class ArXivBERTData(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        labels = np.array(labels)
        
        encodings = transform(data)

        p_data_idx = np.where(labels==1)[0]
        n_data_idx = np.where(labels==0)[0]
        
        self.p_data = encodings[p_data_idx, :, :]
        self.n_data = encodings[n_data_idx, :, :]

        self.labels = labels

        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.labels)

class ParamveerTestData(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        labels = np.array(labels)
        
        encodings = transform(data)

        u_p_data_idx = np.where(labels==-1)[0]
        p_p_data_idx = np.where(labels==1)[0]
        n_data_idx = np.where(labels==0)[0]
        
        self.p_data = encodings[p_p_data_idx, :, :]
        self.n_data = encodings[n_data_idx, :, :]
        self.u_p_data = encodings[u_p_data_idx, :, :]

        self.labels = labels

        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.labels)