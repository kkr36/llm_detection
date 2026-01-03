from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast, RobertaTokenizerFast
import pandas as pd
import json
import spacy
from tqdm import tqdm
import re

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir=="neg" else 1)

    return texts, labels

def read_semeval_split(split_dir, split):
    if "test_sample" in split:
        load_split = "test_sample"
    else:
        load_split = split        
    data = pd.read_parquet(f"{split_dir}/{load_split}.parquet")

    if "front" in split:
        data = data.iloc[:int(len(data)*.7)]
    elif "back" in split:
        data = data.iloc[int(len(data)*.7):]
    print(f"{split} : loaded {len(data)} samples; {sum(data['label'])} pos {len(data) - sum(data['label'])} neg")

    # if split=="validation":
    #     data = data.iloc[-100:]
    # if split=="train":
    #     data = data.iloc[-2000:]
    # if split == "train":
    #     subset = data.iloc[:int(len(data) * .3)]
    # elif split == "validation":
    #     subset = data.iloc[int(len(data) * .3):int(len(data) * .4)]
    
    texts, labels = data["code"].tolist(), data["label"].tolist()
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


def getCodeBertTokenizer(model):
    assert model == "microsoft/codebert-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model)
    return tokenizer


# def initialize_codebert_transform(net):
#     assert(net == "microsoft/codebert-base")
#     tokenizer = getCodeBertTokenizer(net)

#     def transform(text):
#         tokens = tokenizer(
#             text,
#             padding=True,
#             truncation=True
#         )

#         # CodeBERT (RoBERTa) → NO token_type_ids
#         x = np.stack(
#             (
#                 tokens["input_ids"],
#                 tokens["attention_mask"],
#             ),
#             axis=2
#         )
#         return x

#     return transform


def initialize_codebert_transform(net, batch_size=4096):
    assert net == "microsoft/codebert-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(net)

    def transform(text_list):
        outputs = []

        for i in tqdm(
            range(0, len(text_list), batch_size),
            total=(len(text_list) + batch_size - 1) // batch_size,
            desc="Tokenizing CodeBERT",
        ):
            batch = text_list[i:i + batch_size]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True
            )

            x = np.stack(
                (tokens["input_ids"], tokens["attention_mask"]),
                axis=2
            )
            outputs.append(x)

        return np.concatenate(outputs, axis=0)

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