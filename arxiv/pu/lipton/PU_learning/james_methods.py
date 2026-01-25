import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
from tqdm import tqdm

from scipy.optimize import minimize
import swifter

import spacy
import re
import json
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