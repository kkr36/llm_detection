import os
import argparse
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score

font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20
    }
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rc('font', **font)

from algorithm import * 
from model_helper import * 
from helper import *
from estimator import *
from baselines import *
import random
from james_methods import *
from data_helper import *

data_dir = '/Users/kkr36/Downloads'

def get_james_save_str(data_type, year, alpha, combine, sentence, clean, add, gemini, flip, split):
    combine_str = '_combine' if combine else ''
    sentence_str = '_sentence' if sentence else ''
    clean_str = '_clean' if clean else ''
    add_str = '_add' if add else ''
    gemini_str = '_gemini' if gemini else ''
    flip_str = '_flip' if flip else ''
    return f"arxiv_tokenized_{split}_{year}_{alpha}_{data_type}{combine_str}{sentence_str}{clean_str}{add_str}{gemini_str}{flip_str}.parquet"

def read_arxiv_unlabeled(test_alpha, test_year, sentence, clean, split):
    arxiv_data = pd.read_parquet(f"{data_dir}/multillm/data_raw/arxiv_{test_year}_ai_cs._10000_fronthalf.parquet")

    llm_cols, llm_writing = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"], []
    for i in tqdm(list(i for i in range(len(arxiv_data)))):
        assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
        original_rewrite = arxiv_data.iloc[i][llm_cols[i % 4]]
        llm_writing.append(original_rewrite)
    arxiv_data['ai_abstract'] = llm_writing

    num_inject = int(.2 * len(arxiv_data))

    right_labels = arxiv_data.iloc[num_inject:].reset_index(drop=True)
    right_labels = right_labels.sample(frac=1, random_state=seed).reset_index(drop=True)
    if split == "out":
        right_labels = right_labels.iloc[5700:6000].reset_index(drop=True)
    else:
        right_labels = right_labels.iloc[2500:6000-1000].reset_index(drop=True) # this should not be seen by any models

    llm_subset = right_labels.iloc[:int(len(right_labels)*test_alpha)]
    human_subset = right_labels.iloc[int(len(right_labels)*test_alpha):]
    # texts, labels = llm_subset["ai_abstract"].tolist() + human_subset["human_abstract"].tolist(), [1 for _ in range(len(llm_subset))] + [0 for _ in range(len(human_subset))]

    if sentence:
        print("splitting human sentences, unlabeled")
        human_texts, _ = split_into_sentences(human_subset["human_abstract"].tolist(), [0 for _ in range(len(human_subset))])
        print("splitting llm sentences, unlabeled")
        llm_texts, _ = split_into_sentences(llm_subset["ai_abstract"].tolist(), [0 for _ in range(len(llm_subset))])

        random.seed(42)
        random.shuffle(human_texts)
        random.shuffle(llm_texts)

        H = len(human_texts)
        L = len(llm_texts)

        l = min(L, math.floor(test_alpha * (H + L)))
        h = math.floor((1 - test_alpha) * (H + L))

        # adjust if we exceed available data
        total = l + h
        scale = min(L / l if l else 1, H / h if h else 1)

        l = int(l * scale)

        h = int(h * scale)

        llm_texts   = llm_texts[:l]
        human_texts = human_texts[:h]
        texts, labels = llm_texts + human_texts, [1 for _ in range(l)] + [0 for _ in range(h)]

    else:
        texts, labels = human_subset["human_abstract"].tolist() + llm_subset["ai_abstract"].tolist(), [0 for _ in range(len(human_subset))] + [1 for _ in range(len(llm_subset))]

    if clean:
        texts = clean_text(texts)

    return texts, labels

def read_arxiv_positive(test_year, sentence, clean, seed):
    
    arxiv_data = pd.read_parquet(f"{data_dir}/multillm/data_raw/arxiv_{test_year}_ai_cs._10000_fronthalf.parquet")

    llm_cols, llm_writing = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"], []
    for i in tqdm(list(i for i in range(len(arxiv_data)))):
        assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
        original_rewrite = arxiv_data.iloc[i][llm_cols[i % 4]]
        llm_writing.append(original_rewrite)
    arxiv_data['ai_abstract'] = llm_writing

    num_inject = int(.2 * len(arxiv_data))

    right_labels = arxiv_data.iloc[num_inject:].reset_index(drop=True)
    right_labels = right_labels.sample(frac=1, random_state=seed).reset_index(drop=True)
    right_labels = right_labels.iloc[:999] # the first 1k should be seen by train set of all models 

    texts = right_labels["ai_abstract"].tolist()

    if sentence:
        print("splitting llm sentences, positive")
        texts, _ = split_into_sentences(texts, [1 for _ in range(len(texts))])

    if clean:
        texts = clean_text(texts)

    return texts

def get_p_data(data_type, test_year, sentence, clean, combine, gemini, flip, seed):
    if data_type == "ArXiv_BERT":
        p_texts = []
        years = [2016, 2018, 2020] if combine else [test_year]
        for year in years:
            p_texts += read_arxiv_positive(year, sentence, clean, seed)

    # turn into dataloader
    transform = initialize_bert_transform('distilbert-base-uncased')

    train_dataset = IMDbBERTData(p_texts, [1 for _ in range(len(p_texts))], transform=transform)
    p_data = PosData(transform=train_dataset.transform, \
                    target_transform=train_dataset.target_transform, \
                    data=train_dataset.p_data, index=np.array(range(len(train_dataset.p_data))), data_type=data_type)
    p_data_loader = torch.utils.data.DataLoader(p_data, batch_size=16, \
        shuffle=False)
    return p_data_loader

def get_u_data(data_type, test_alpha, test_year, combine, sentence, clean, add, gemini, flip, split, seed):
    if data_type == "ArXiv_BERT":
        u_texts, u_labels = [], []
        years = [2016, 2018, 2020] if combine else [test_year]
        for year in years:
            ut, ul = read_arxiv_unlabeled(test_alpha, year, sentence, clean, split, seed)
            u_texts += ut
            u_labels += ul

    pu, nu = sum(u_labels), len(u_labels) - sum(u_labels)
    assert(round(pu / (pu+nu), 2) == test_alpha)    

    # turn into dataloader
    transform = initialize_bert_transform('distilbert-base-uncased')

    train_dataset = IMDbBERTData(u_texts, u_labels, transform=transform)
    u_data = UnlabelData(transform=train_dataset.transform, \
                target_transform=train_dataset.target_transform, \
                pos_data=train_dataset.p_data, neg_data=train_dataset.n_data, \
                index=np.array(range(len(u_texts))),data_type=data_type)
    u_data_loader = torch.utils.data.DataLoader(u_data, batch_size=16, \
        shuffle=False)
    return u_data_loader, u_texts, u_labels

def get_p_data_llm(data_type, test_year, sentence, clean, test_llm, gemini, flip, seed):

    def read_arxiv_positive_llm(test_year, test_llm, sentence, clean, flip, seed):
        llm_cols = ["Llama 3.3 70b Instruct", "GPT OSS 120b", "Gemini 2.5 Flash", "Gemini 3 Preview"] if not gemini else ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"]
        assert(test_llm in llm_cols or test_llm=="all"), f"{test_llm} not valid"
        
        arxiv_data = pd.read_parquet(f'{data_dir}/multillm/data_raw/arxiv_{test_year}_ai_cs._10000_fronthalf.parquet' if not gemini else f"{data_dir}/multillm/data_raw/arxiv_{test_year}_ai_cs._10000_fronthalf_gemini_full.parquet")

        llm_subset = arxiv_data[arxiv_data[test_llm].notna() & (arxiv_data[test_llm] != "")].reset_index(drop=True) # isolate llm writing

        # shuffle
        llm_subset = llm_subset.sample(frac=1, random_state=seed).reset_index(drop=True)
        llm_subset = llm_subset.iloc[int(len(llm_subset)*.75):]

        llm_texts = llm_subset[test_llm if test_llm != "all" else "llm_writing"].tolist()
        human_texts = llm_subset['human_abstract'].tolist()
        assert(len(llm_texts) == len(human_texts))

        if flip:
            # import pdb; pdb.set_trace()
            positive_texts = human_texts[int(len(human_texts)*.75):]
        else:
            positive_texts = llm_texts[int(len(llm_texts)*.75):]

        # sentence check
        if sentence:
            
            positive_texts, _ = split_into_sentences(positive_texts, [1 for _ in range(len(positive_texts))])
        
        if clean:
            positive_texts = clean_text(positive_texts)

        return positive_texts


    if data_type == "ArXiv_BERT":
        p_texts = read_arxiv_positive_llm(test_year, test_llm, sentence, clean, flip, seed)

    # turn into dataloader
    transform = initialize_bert_transform('distilbert-base-uncased')

    train_dataset = IMDbBERTData(p_texts, [1 for _ in range(len(p_texts))], transform=transform)
    p_data = PosData(transform=train_dataset.transform, \
                    target_transform=train_dataset.target_transform, \
                    data=train_dataset.p_data, index=np.array(range(len(train_dataset.p_data))), data_type=data_type)
    p_data_loader = torch.utils.data.DataLoader(p_data, batch_size=16, \
        shuffle=False)
    return p_data_loader

def get_u_data_llm(data_type, test_alpha, test_year, test_llm, sentence, clean, gemini, flip, split, seed):

    def read_arxiv_unlabeled_llm(test_alpha, test_year, test_llm, sentence, clean, flip, split, seed):
        llm_cols = ["Llama 3.3 70b Instruct", "GPT OSS 120b", "Gemini 2.5 Flash", "Gemini 3 Preview"] if not gemini else ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"]
        assert(test_llm in llm_cols or test_llm=="all"), f"{test_llm} not valid"
        
        arxiv_data = pd.read_parquet(f'{data_dir}/multillm/data_raw/arxiv_{test_year}_ai_cs._10000_fronthalf.parquet' if not gemini else f"{data_dir}/multillm/data_raw/arxiv_{test_year}_ai_cs._10000_fronthalf_gemini_full.parquet")

        llm_subset = arxiv_data[arxiv_data[test_llm].notna() & (arxiv_data[test_llm] != "")].reset_index(drop=True) # isolate llm writing

        # shuffle
        llm_subset = llm_subset.sample(frac=1, random_state=seed).reset_index(drop=True)
        llm_subset = llm_subset.iloc[int(len(llm_subset)*.75):]

        llm_texts = llm_subset[test_llm if test_llm != "all" else "llm_writing"].tolist()
        human_texts = llm_subset['human_abstract'].tolist()
        assert(len(llm_texts) == len(human_texts))

        if flip:
            u_positive_texts = human_texts[:int(len(human_texts)*.75)]
            u_negative_texts = llm_texts[:int(len(llm_texts)*.75)]
        else:
            u_positive_texts = llm_texts[:int(len(llm_texts)*.75)]
            u_negative_texts = human_texts[:int(len(human_texts)*.75)]

        # sentence check
        if sentence:
            u_positive_texts, _ = split_into_sentences(u_positive_texts, [1 for _ in range(len(u_positive_texts))])
            u_negative_texts, _ = split_into_sentences(u_negative_texts, [0 for _ in range(len(u_negative_texts))])

        # Compute feasible T bounds
        T_pos = len(u_positive_texts) / test_alpha if test_alpha > 0 else np.inf
        T_neg = len(u_negative_texts) / (1 - test_alpha) if test_alpha < 1 else np.inf

        T = int(min(T_pos, T_neg))

        n_pos = int(test_alpha * T)
        n_neg = T - n_pos  # ensures n_pos + n_neg = T exactly

        u_positive_texts = list(np.random.default_rng(42).choice(u_positive_texts, size=n_pos, replace=False))
        u_negative_texts = list(np.random.default_rng(42).choice(u_negative_texts, size=n_neg, replace=False))
        print(f"alpha = {len(u_positive_texts)} / {len(u_positive_texts) + len(u_negative_texts)} = {len(u_positive_texts) / (len(u_positive_texts) + len(u_negative_texts))}")
        
        if clean:
            u_positive_texts = clean_text(u_positive_texts)
            u_negative_texts = clean_text(u_negative_texts)

        return u_positive_texts + u_negative_texts, [1 for _ in range(len(u_positive_texts))] + [0 for _ in range(len(u_negative_texts))]

    if data_type == "ArXiv_BERT":
        u_texts, u_labels = read_arxiv_unlabeled_llm(test_alpha, test_year, test_llm, sentence, clean, flip, split, seed)

    pu, nu = sum(u_labels), len(u_labels) - sum(u_labels)
    assert(round(pu / (pu+nu), 2) == test_alpha)    

    # turn into dataloader
    transform = initialize_bert_transform('distilbert-base-uncased')

    train_dataset = IMDbBERTData(u_texts, u_labels, transform=transform)
    u_data = UnlabelData(transform=train_dataset.transform, \
                target_transform=train_dataset.target_transform, \
                pos_data=train_dataset.p_data, neg_data=train_dataset.n_data, \
                index=np.array(range(len(u_texts))),data_type=data_type)
    u_data_loader = torch.utils.data.DataLoader(u_data, batch_size=16, \
        shuffle=False)
    return u_data_loader, u_texts, u_labels

def get_preds(data_type, net, device, test_alpha, test_year, combine, sentence, clean, add, gemini, flip, seed):
    p_data_loader = get_p_data(data_type, test_year, sentence, clean, combine, gemini, flip, seed)
    u_data_loader, _, _ = get_u_data(data_type, test_alpha, test_year, combine, sentence, clean, add, gemini, flip, "in", seed)

    # get preds with model
    pos_probs = p_probs(net, device, p_data_loader)
    unlabeled_probs, unlabeled_targets = u_probs(net, device, u_data_loader)

    # return pos preds, u preds, u labels
    return pos_probs, unlabeled_probs, unlabeled_targets

def get_preds_llm(data_type, net, device, test_alpha, test_year, test_llm, sentence, clean, gemini, flip, seed):
    p_data_loader = get_p_data_llm(data_type, test_year, sentence, clean, test_llm, gemini, flip, seed)
    u_data_loader, _, _ = get_u_data_llm(data_type, test_alpha, test_year, test_llm, sentence, clean, gemini, flip, "in", seed)

    # get preds with model
    pos_probs = p_probs(net, device, p_data_loader)
    unlabeled_probs, unlabeled_targets = u_probs(net, device, u_data_loader)

    # return pos preds, u preds, u labels
    return pos_probs, unlabeled_probs, unlabeled_targets

def tokenize_fn(data_type, year, alpha, combine, sentence, clean, add, gemini, flip, split):
    assert(sentence)
    u_texts, u_labels = [], []
    years = [2016, 2018, 2020] if combine else [year]
    print(years)

    for year_tmp in years:
        if split == "val":
            _, u_texts_tmp, u_labels_tmp = get_u_data(data_type, alpha, year_tmp, combine, sentence, clean, add, gemini, flip)
        elif split == "train":
            split_dir = f'{data_dir}/multillm/double_rewrite/arxiv_{year_tmp}_ai_cs._10000_0.2_fronthalf.parquet'
            if add:
                u_texts_tmp, u_labels_tmp = read_arxiv_split_add(split_dir, alpha, "train", sentence, inject=True)
            else:
                u_texts_tmp, u_labels_tmp = read_arxiv_split2(split_dir, alpha, "train", sentence, inject=True)
        u_texts += u_texts_tmp
        u_labels += u_labels_tmp

    n_texts = [t for t, y in zip(u_texts, u_labels) if y == 0]
    p_texts = [t for t, y in zip(u_texts, u_labels) if y == 1]

    tokenized = defaultdict(list)

    # tokenize human/ai abs separately
    for i, ai_abstract in tqdm(list(enumerate(p_texts))):
        # tokenized_abs = tokenize(ai_abstract)
        words = re.findall(r'\b\w+\b', ai_abstract.lower())
        # Remove any words that are numeric
        words_without_digits=[word for word in words if not word.isdigit()]
        tokenized['ai_sentence'].append(words_without_digits)
        tokenized['ai_index'].append(i)
        # import pdb; pdb.set_trace()
    for i, human_abstract in tqdm(list(enumerate(n_texts))):
        # tokenized_abs = tokenize(human_abstract)
        words = re.findall(r'\b\w+\b', human_abstract.lower())
        # Remove any words that are numeric
        words_without_digits=[word for word in words if not word.isdigit()]
        tokenized['human_sentence'].append(words_without_digits)
        tokenized['human_index'].append(i)

    # tokenized abs --> parquet --> save

    # Find max number of rows
    max_len = max(len(v) for v in tokenized.values())

    # Pad shorter columns with None
    for k, v in tokenized.items():
        if len(v) < max_len:
            if "index" in k:
                for _ in range(max_len - len(v)):
                    v.append(-1)
            else:
                v.extend([['']] * (max_len - len(v)))

    # Make DataFrame
    df = pd.DataFrame(tokenized)

    # Save to Parquet
    # if not os.path.exists(f"{data_dir}/multillm/james_v_us"):
    #     os.makedirs(f"{data_dir}/james_v_us")
    file_str = get_james_save_str(data_type, year, alpha, combine, sentence, clean, add, gemini, flip, split)
    save_path = f"{data_dir}/multillm/james_v_us/{file_str}"
    df.to_parquet(save_path, index=False)

def estimate_train(data_type, train_year, alpha, combine, sentence, clean, add, gemini, flip, split):
    assert(split=="train")
    load_str = get_james_save_str(data_type, train_year, alpha, combine, sentence, clean, add, gemini, flip, split)
    train_data = pd.read_parquet(f"{data_dir}/multillm/james_v_us/{load_str}")

    human_data = train_data[["human_sentence", "human_index"]]
    human_train = human_data[human_data['human_index'] != -1]
    ai_data = train_data[["ai_sentence", "ai_index"]]
    ai_train = ai_data[ai_data['ai_index'] != -1]

    estimate_text_distribution(human_train, ai_train, f"{data_dir}/multillm/james_v_us/{load_str.replace('.parquet', '_logprob.parquet')}")

def MLE_james(data_type, train_year, alpha, test_year, test_alpha, combine, sentence, clean, add, gemini, flip, test_cis, n_bootstrap):
    def df_to_val_set(df, t_alpha):
        human_data = df[["human_sentence", "human_index"]]
        human_subset = human_data[human_data['human_index'] != -1]
        ai_data = df[["ai_sentence", "ai_index"]]
        ai_subset = ai_data[ai_data['ai_index'] != -1]

        ai_subset["inference_sentence"] = ai_subset["ai_sentence"]
        human_subset["inference_sentence"] = human_subset["human_sentence"]
        return pd.concat([ai_subset[["inference_sentence"]], human_subset[["inference_sentence"]]]) if t_alpha != 0 else pd.DataFrame(human_subset["inference_sentence"]) if t_alpha != 1 else pd.DataFrame(ai_subset["inference_sentence"])

    logprob_str = get_james_save_str(data_type, train_year, alpha, combine, sentence, clean, add, gemini, flip, "train")
    logprob_path = f"{data_dir}/multillm/james_v_us/{logprob_str.replace('.parquet', '_logprob.parquet')}"
    mle = MLE(logprob_path)
    val_data_str = get_james_save_str(data_type, test_year, test_alpha, combine, sentence, clean, add, gemini, flip, "val")
    val_data_path = f"{data_dir}/multillm/james_v_us/{val_data_str}"
    val_data = pd.read_parquet(val_data_path)
    val_data = df_to_val_set(val_data, test_alpha)
    solution, half_widths = mle.inference(val_data,True,n_bootstrap,test_cis)
    return solution, half_widths