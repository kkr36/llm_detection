from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import pandas as pd
import json
import spacy
from tqdm import tqdm

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
        ft_data, ft_labels = ft_data[:int(n_ft * .8)], ft_labels[:int(n_ft * .8)]

    elif split=="val":
        ai_data, ai_labels = ai_data[int(n_ai * .8):], ai_labels[int(n_ai * .8):]
        human_data, human_labels = human_data[int(n_human * .8):], human_labels[int(n_human * .8):]
        ft_data, ft_labels = ft_data[int(n_ft * .8):], ft_labels[int(n_ft * .8):]

    elif split=="test":
        pos_ai_data = ai_data[int(n_ai*.7):int(n_ai*.8)]
        u_ai_data = ai_data[int(n_ai*.8):]
        ft_data = ft_data[int(n_ft * .8):]

        if ft:
            return ft_data + pos_ai_data, [0 for _ in range(len(ft_data))] + [1 for _ in range(len(pos_ai_data))]
        else:
            return u_ai_data + pos_ai_data, [0 for _ in range(len(u_ai_data))] + [1 for _ in range(len(pos_ai_data))]
    
    # train/val
    if ft:
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

def read_arxiv_split2(split_dir, split="train"):
    arxiv_data = pd.read_parquet(split_dir)
    llm_writing = []
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
    for i in (list(i for i in range(len(arxiv_data)))):
        assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
        llm_writing.append(arxiv_data.iloc[i][llm_cols[i % 4]])
    arxiv_data['ai_abstract'] = llm_writing
    assert(split in ["train", "val"])
    if split == "train":
        subset = arxiv_data.iloc[:int(len(arxiv_data)*.75)]
    elif split == "val":
        subset = arxiv_data.iloc[int(len(arxiv_data)*.75):]
    texts = subset["human_abstract"].tolist() + subset["ai_abstract"].tolist()
    labels = [0 for _ in range(len(subset))] + [1 for _ in range(len(subset))]
    # import pdb; pdb.set_trace()

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
