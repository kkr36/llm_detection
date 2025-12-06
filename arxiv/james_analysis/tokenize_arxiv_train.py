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
    Processes the input text, splits it into sentences, and further processes each sentence
    to extract non-numeric words. It constructs a list of these words for each sentence.

    Parameters:
    text (str): A string containing multiple sentences.

    Returns:
    list: A list of lists, where each inner list contains the words from one sentence,
          excluding any numeric strings.
    """
    # remove newline characters, this line is not necessary for all cases
    # the reason it is included here is because the abstracts in the dataset contain abnormal newline characters
    # e.g. Recent works on diffusion models have demonstrated a strong capability for\nconditioning image generation,
    text=text.replace('\n',' ')
    # Initialize an empty list to store the list of words for each sentence
    sentence_list=[]
    # Process the sentence using the spacy model to extract linguistic features and split into components
    doc=nlp(text)
    # Iterate over each sentence in the processed text
    for sent in doc.sents:
        # Extract the words from the sentence
        words = re.findall(r'\b\w+\b', sent.text.lower())
        # Remove any words that are numeric
        words_without_digits=[word for word in words if not word.isdigit()]
        # If the list is not empty, append the list of words to the sentence_list
        if len(words_without_digits)!=0:
            sentence_list.append(words_without_digits)
    return sentence_list

if __name__ == "__main__":
    years = range(2010,2026,1)
    # years = [2024]
    # years = [2014, 2015, 2016, 2017, 2018, 2019]
    # years = [2021,2022]
    # years = [2023]
    years = list(range(2010,2026))
    subsample_size = 5000
    # subsample_size = 2500
    category = 'cs.'
    for year in tqdm(years):
        # load data
        tokenized = defaultdict(list)
        arxiv_path = f"/share/garg/arxiv_kaggle/train/arxiv-metadata-oai-snapshot_{year}_ai_{category}_{subsample_size}.json"
        if not os.path.exists(arxiv_path):
            arxiv_path = f"/share/garg/arxiv_kaggle/train/arxiv-metadata-oai-snapshot_{year}_ai_{category}_{subsample_size//2}.json"
        with open(arxiv_path, 'r') as f:
            arxiv_data = json.load(f)

        # tokenize human/ai abs separately
        assert(len(arxiv_data['ai_abs']) >= 2500 and len(arxiv_data['human_abs']) >= 2500)
        for i, ai_abstract in tqdm(list(enumerate(arxiv_data['ai_abs'][:2500]))):
            tokenized_abs = tokenize(ai_abstract)
            tokenized['ai_sentence'] += tokenized_abs
            tokenized['ai_index'] += [i for _ in range(len(tokenized_abs))]
            # import pdb; pdb.set_trace()
        for i, human_abstract in tqdm(list(enumerate(arxiv_data['human_abs'][:2500]))):
            tokenized_abs = tokenize(human_abstract)
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
        save_path = f"/share/garg/arxiv_kaggle/train/arxiv_tokenized_{year}_ai_{category}_{subsample_size//2}.parquet"
        df.to_parquet(save_path, index=False)
        print(f"{year} saved to {save_path}")
