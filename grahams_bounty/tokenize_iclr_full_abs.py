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
    # years = list(range(2020,2027))
    years = [2026]
    # years = list(range(2020,2026))

    # load data
    # iclr_path = f"/share/garg/openreview_data/graham/iclr-dataset/data/iclr26v1.parquet"
    iclr_path = "2026_with_ratings.parquet"

    iclr_data_all = pd.read_parquet(iclr_path)
    import pdb; pdb.set_trace()
    
    for year in years:
        tokenized = defaultdict(list)

        iclr_data = iclr_data_all[iclr_data_all["year"] == year]

        for i, iclr_abstract in tqdm(list(enumerate(iclr_data['abstract'].tolist()[:]))):
            # import pdb; pdb.set_trace()
            keywords = iclr_data.iloc[i]['keywords']
            tokenized_abs = tokenize(iclr_abstract)
            full_abs = [token for sentence in tokenized_abs for token in sentence]
            tokenized['human_abstract'].append(full_abs)
            tokenized['keywords'].append(keywords)

            tokenized['scores'].append(iclr_data.iloc[i]['scores'])

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
        save_path = f"/share/garg/openreview_data/graham/iclr-dataset/data/tokenized_iclr_{year}_full_abs.parquet"
        df.to_parquet(save_path, index=False)
