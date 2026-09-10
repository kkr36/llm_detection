import re
import string
import numpy as np
from typing import Dict, List
import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import spacy
nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
nlp.enable_pipe("senter")
import time

def compute_abstract_stats(data: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    For each key in the dictionary:
        - Computes average number of words per abstract
        - Computes average number of words per sentence
        - Computes average lexical burstiness per abstract
    """

    results = {}

    sentence_splitter = re.compile(r'[.!?]+')
    punctuation_table = str.maketrans('', '', string.punctuation)

    for key, abstracts in data.items():
        total_words = 0
        total_sentences = 0
        total_abstracts = len(abstracts)

        burstiness_scores = []

        for abstract in abstracts:
            # ----- Normalize text -----
            cleaned = abstract.lower().translate(punctuation_table)
            words = cleaned.split()
            num_words = len(words)

            if num_words == 0:
                continue

            total_words += num_words

            # ----- Sentence count -----
            sentences = [
                s.strip() for s in sentence_splitter.split(abstract) if s.strip()
            ]
            total_sentences += len(sentences)

            # ----- Burstiness heuristic -----
            unique_words = len(set(words))
            burstiness = 1 - (unique_words / num_words)
            burstiness_scores.append(burstiness)

        avg_words_per_abstract = (
            total_words / total_abstracts if total_abstracts > 0 else 0.0
        )

        avg_words_per_sentence = (
            total_words / total_sentences if total_sentences > 0 else 0.0
        )

        avg_burstiness = (
            float(np.mean(burstiness_scores)) if burstiness_scores else 0.0
        )

        std_burstiness = (
            float(np.std(burstiness_scores)) if burstiness_scores else 0.0
        )

        results[key] = {
            "avg_words_per_abstract": avg_words_per_abstract,
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_burstiness": avg_burstiness,
            "std_burstiness": std_burstiness,
        }

    return results


def split_into_sentences(abstract):
    try:
        doc = nlp(abstract)
    except:
        print(abstract, "bad")
        doc = nlp("")
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def clean_text(text):
    # replace bad things we know how to; remove all other non-typable
    bad_dashes = ['—', '⁻', '–', '‑', '‐', '−']
    bad_apostrophes = ['’', '′', '‘']
    bad_left_quote = "“"
    bad_right_quote = "”"
    for bad_dash in bad_dashes:
        text = [t.replace(bad_dash, '-') for t in text]
    for bad_apostrophe in bad_apostrophes:
        text = [t.replace(bad_apostrophe, "'") for t in text]
    text = [t.replace(bad_left_quote, '"') for t in text]
    text = [t.replace(bad_right_quote, '"') for t in text]

    # text = [unidecode(t) for t in text]
    text = [re.sub(r"[\n\t]+", " ", t) for t in text] # consecutive tabs, newline
    # text = [re.sub(r"[ ]+", " ", t) for t in text] # consecutive spaces
    # text = [t.strip() for t in text] # leading, trailing whitespace

    # remove non-typable
    allowed = set(string.printable)
    text = [''.join(ch for ch in s if ch in allowed) for s in text]
    return text

def pretty(i, data): return "\n".join(f"{key}: {data[key][i]}" for key in data)

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

def individual_predict(net, device, sample):
    if isinstance(sample, str):
        # transform to arr
        transform = initialize_bert_transform('distilbert-base-uncased')
        sample = transform([sample])
    input = torch.from_numpy(sample)
    net.eval()
    with torch.no_grad():
        input = input.to(device)
        output = net(input)
        probs = torch.nn.functional.softmax(output, dim=-1)[:,0] 
    return output, probs

def predict_with_backoff(pangram_client, text, max_retries=5, initial_delay=1):
    """
    Call pangram_client.predict with exponential backoff retry logic.
    
    Args:
        pangram_client: The Pangram client instance
        text: Text to predict on
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles with each retry)
    
    Returns:
        API response dict or None if all retries failed
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            result = pangram_client.predict(text)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            
            # Exponential backoff
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Double the delay for next attempt
    
    return {}