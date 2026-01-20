import spacy
import re
import os
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pdb
nlp = spacy.load("en_core_web_lg")
print("loaded nlp")

def tokenize(text):
    """
    Processes the input text, splits it into logical lines, and further processes each line
    to extract non-numeric words. It constructs a list of these words for each line.

    Parameters:
    text (str): A string containing code or text with line breaks.

    Returns:
    list: A list of lists, where each inner list contains the words from one line,
          excluding any numeric strings.
    """
    # Split into logical lines and drop empty / whitespace-only lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    line_list = []

    for line in lines:
        # Run the same spaCy pipeline for consistency
        doc = nlp(line)

        # Extract words using the same rule as before
        words = re.findall(r'\b\w+\b', doc.text.lower())

        # Remove numeric-only tokens
        words_without_digits = [w for w in words if not w.isdigit()]

        if words_without_digits:
            line_list.append(words_without_digits)

    # import pdb; pdb.set_trace()
    return line_list

from itertools import chain

def tokenize_batch(texts):
    """
    Batch process multiple texts efficiently.
    
    Parameters:
    texts (list): List of strings to process
    
    Returns:
    list: List of results, one per input text
    """
    results = []
    
    # Prepare all lines from all texts with tracking info
    all_lines = []
    line_to_text_idx = []  # Track which text each line belongs to
    text_line_counts = []  # Track how many lines per text
    
    for text_idx, text in enumerate(texts):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        all_lines.extend(lines)
        line_to_text_idx.extend([text_idx] * len(lines))
        text_line_counts.append(len(lines))
    
    # Process all lines in one batch
    docs = list(tqdm(nlp.pipe(all_lines, batch_size=100, n_process=4), 
                     total=len(all_lines),
                     desc="Processing lines"))
        
    # Extract words from all docs
    all_processed_lines = []
    for doc in docs:
        words = re.findall(r'\b\w+\b', doc.text.lower())
        words_without_digits = [w for w in words if not w.isdigit()]
        all_processed_lines.append(words_without_digits if words_without_digits else None)
    
    # Group results back by original text
    line_idx = 0
    for count in text_line_counts:
        line_list = [
            words for words in all_processed_lines[line_idx:line_idx + count]
            if words is not None
        ]
        results.append(line_list)
        line_idx += count
    
    return results

if __name__ == "__main__":

    splits = ["train", "validation", "test_sample"][:1]
    data_sizes = [6000, 3000, 1000][:1]

    for split, data_size in zip(splits, data_sizes):
        # load data
        arxiv_path = f"/share/garg/kkr36/Task_A/{split}.parquet"
        arxiv_data = pd.read_parquet(arxiv_path)
        subset = arxiv_data.sample(data_size,random_state=42,replace=False)

        tokenized = defaultdict(list)

        human_writing = subset[subset['label'] == 0]['code']
        llm_writing = subset[subset['label'] == 1]['code']

        human_writing, llm_writing = tokenize_batch(human_writing), tokenize_batch(llm_writing)
        
        # tokenize human/ai abs separately
        for i, tokenized_abs in tqdm(list(enumerate(llm_writing))):
            # tokenized_abs = tokenize(ai_abstract)
            tokenized['ai_sentence'] += tokenized_abs
            tokenized['ai_index'] += [i for _ in range(len(tokenized_abs))]
            # import pdb; pdb.set_trace()
        for i, tokenized_abs in tqdm(list(enumerate(human_writing))):
            # tokenized_abs = tokenize(human_abstract)
            tokenized['human_sentence'] += tokenized_abs
            tokenized['human_index'] += [i for _ in range(len(tokenized_abs))]

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
        # import pdb; pdb.set_trace()

        # Save to Parquet
        save_path = f"/share/garg/kkr36/Task_A/{split}_tokenized_{data_size}.parquet"
        df.to_parquet(save_path, index=False)
        print(f"{split} saved to {save_path}")
