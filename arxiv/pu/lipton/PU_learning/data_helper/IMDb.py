from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import pandas as pd
import json
import spacy
from tqdm import tqdm
import re
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.enable_pipe("senter")

import math

def split_into_sentences(abstracts, labels):
    all_sentences = []
    all_labels = []
    print("Splitting into sentences!")
    for i, abstract in tqdm(list(enumerate(abstracts))):
        doc = nlp(abstract)
        sentences = [sent.text.strip() for sent in doc.sents]
        sentences = [s[i:i+len(s)//2 + len(s)%2] for s in sentences for i in (0, len(s)//2 + len(s)%2)]

        # sentences = sentences[:2] + sentences[-2:]
        all_sentences.extend(sentences)  # extend instead of append
        all_labels.extend([labels[i] for _ in range(len(sentences))])
    return all_sentences, all_labels

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.enable_pipe("senter")

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

def read_arxiv_split2(split_dir, alpha=.6, split="train", sentence=False, inject=True):
    # import os
    # if not(os.path.exists(split_dir)):
    year = int(re.search(r"\d{4}", split_dir).group(0))
    pct_inject = 0 if (year < 2014 or inject is False) else 0.05 * (year - 2012) / 2
    # pct_inject = 0 # TODO remove
    
    arxiv_data = pd.read_parquet(split_dir)
    # tmp = arxiv_data.copy(deep=True)
    num_inject = int(pct_inject * len(arxiv_data))
    print(f"injecting {num_inject} LLM-written abstracts ({pct_inject}) into pool of {len(arxiv_data)} abstracts")

    llm_writing = []
    inject_counter = num_inject
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
    for i in tqdm(list(i for i in range(len(arxiv_data)))):
        assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
        original_rewrite = arxiv_data.iloc[i][llm_cols[i % 4]]
        original_rewrite = "THIS IS AI WRITING -- I'm Watermarking here!!"
        # import pdb; pdb.set_trace()
        # if inject_counter > 0: 
        #     import pdb; pdb.set_trace()
        if inject_counter > 0 and arxiv_data.iloc[i][llm_cols[(i+1)%4]] is None:
            import pdb; pdb.set_trace()
        if inject_counter > 0 and len(arxiv_data.iloc[i][llm_cols[(i+1)%4]]) > 0:
            if not inject:
                assert(False), "should not be injecting"
            mirror = arxiv_data.iloc[i][llm_cols[(i+1)%4]]
            mirror = "THIS IS AI WRITING -- I'm Watermarking here!!"
            llm_writing.append(mirror)
            arxiv_data.at[i, 'human_abstract'] = original_rewrite
            inject_counter -= 1
        else:
            llm_writing.append(original_rewrite)
    arxiv_data['ai_abstract'] = llm_writing
    assert(inject_counter == 0)

    # import pdb; pdb.set_trace()

    wrong_labels = arxiv_data.iloc[:num_inject].reset_index(drop=True)
    right_labels = arxiv_data.iloc[num_inject:].reset_index(drop=True)

    assert(split in ["train", "val"])
    if split == "train":
        x = int(500 * (3 - 5 * alpha))
        y = 5000 - x
        wrong_labels_subset = wrong_labels.iloc[:int(len(wrong_labels)*.75)-x]
        right_labels_subset = right_labels.iloc[:int(len(right_labels)*.75)-y] # was once just -5000
        subset = pd.concat([wrong_labels_subset, right_labels_subset]).reset_index(drop=True)
        # subset = arxiv_data.iloc[:int(len(arxiv_data)*.75)]
        # subset = wrong_labels.iloc[:int(len(wrong_labels)*.75)]
    elif split == "val":
        x = int(500 - (834 * alpha))
        y = 1666 - x
        wrong_labels_subset = wrong_labels.iloc[int(len(wrong_labels)*.75)+x:]
        right_labels_subset = right_labels.iloc[int(len(right_labels)*.75)+y:] # was once just + 1666
        subset = pd.concat([wrong_labels_subset, right_labels_subset]).reset_index(drop=True)
        # subset = arxiv_data.iloc[int(len(arxiv_data)*.75):]
    # if num_inject > 0:
        # import pdb; pdb.set_trace()

    # print("removing sentences")
    # texts = [remove_first_last_sentence(abs) for abs in texts]
    # print("removed sentences")
    # if split=="val": import pdb; pdb.set_trace()
    if sentence:
        print("splitting into sentences")
        wrong_texts, wrong_labels = split_into_sentences(wrong_labels_subset["human_abstract"].tolist() + wrong_labels_subset["ai_abstract"].tolist(), [0 for _ in range(len(wrong_labels_subset))] + [1 for _ in range(len(wrong_labels_subset))])
        right_texts, right_labels = split_into_sentences(right_labels_subset["human_abstract"].tolist() + right_labels_subset["ai_abstract"].tolist(), [0 for _ in range(len(right_labels_subset))] + [1 for _ in range(len(right_labels_subset))])
        texts, labels = wrong_texts + right_texts, wrong_labels + right_labels
        # import pdb; pdb.set_trace()
        # print(f"Pollution {split}: {sum(wrong_labels)} / {sum(wrong_labels)} + {sum(right_labels)} = {sum(wrong_labels) / (sum(right_labels) + sum(wrong_labels))}")
    
    else:
        texts = subset["human_abstract"].tolist() + subset["ai_abstract"].tolist()
        labels = [0 for _ in range(len(subset))] + [1 for _ in range(len(subset))]
        # print(f"Pollution {split}: {len(wrong_labels_subset)} / {len(wrong_labels_subset)}+ {len(right_labels_subset)} = {len(wrong_labels_subset) / (len(right_labels_subset) + len(wrong_labels_subset))}")

    return texts, labels

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