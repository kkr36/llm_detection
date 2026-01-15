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
    # years = [2020, 2023, 2025]
    years = [2010, 2012, 2014, 2016, 2018, 2020]
    # years = [2024]
    # years = [2014, 2015, 2016, 2017, 2018, 2019]
    # years = list(range(2010,2026))
    subsample_size = 20000//2
    # subsample_size = 2500
    category = 'cs.'
    llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash", "all"][:]
    splits = ["train", "val"]

    for year in tqdm(years):
        arxiv_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size}_fronthalf.parquet"
        arxiv_data = pd.read_parquet(arxiv_path)

        for llm_col in llm_cols[-1:]:
            for split in splits:
                # load data
                tokenized = defaultdict(list)

                if llm_col == 'all':
                    llm_writing = []
                    human_writing = []

                    for llm in llm_cols[:-1]:
                        subset = arxiv_data[arxiv_data[llm].notna() & (arxiv_data[llm] != "")].reset_index(drop=True)
                        llm_writing_tmp, human_writing_tmp = subset[llm].tolist()[:2500//4], subset["human_abstract"].tolist()[:2500//4]
                        if split == "train":
                            llm_writing_tmp = llm_writing_tmp[:int(len(llm_writing_tmp)*.75)]
                            human_writing_tmp = human_writing_tmp[:int(len(human_writing_tmp)*.75)]
                        elif split == "val":
                            llm_writing_tmp = llm_writing_tmp[int(len(llm_writing_tmp)*.75):]
                            human_writing_tmp = human_writing_tmp[int(len(human_writing_tmp)*.75):]
                        llm_writing += llm_writing_tmp
                        human_writing += human_writing_tmp
                else:
                    subset = arxiv_data[arxiv_data[llm_col].notna() & (arxiv_data[llm_col] != "")].reset_index(drop=True)
                    if split == "train":
                        llm_writing, human_writing = subset[llm_col].iloc[:int(len(subset)*.75)].tolist(), subset["human_abstract"].iloc[:int(len(subset)*.75)].tolist()
                    elif split == "val":
                        llm_writing, human_writing = subset[llm_col].iloc[int(len(subset)*.75):].tolist(), subset["human_abstract"].iloc[int(len(subset)*.75):].tolist()

                # import pdb; pdb.set_trace()
                
                # tokenize human/ai abs separately
                for i, ai_abstract in tqdm(list(enumerate(llm_writing))):
                    tokenized_abs = tokenize(ai_abstract)
                    tokenized['ai_sentence'] += tokenized_abs
                    tokenized['ai_index'] += [i for _ in range(len(tokenized_abs))]
                    # import pdb; pdb.set_trace()
                for i, human_abstract in tqdm(list(enumerate(human_writing))):
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
                save_path = f"/share/garg/arxiv_kaggle/multillm/{split}/arxiv_tokenized_{year}_ai_{category}_{subsample_size}_{llm_col}_sentence.parquet"
                df.to_parquet(save_path, index=False)
                print(f"{year} {llm_col} saved to {save_path}")
