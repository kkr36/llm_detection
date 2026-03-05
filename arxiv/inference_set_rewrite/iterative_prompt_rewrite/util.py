import re
import string
import numpy as np
from typing import Dict, List

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


def split_sentences(text: str) -> tuple[str, str]:
    """
    Splits text into the first 2 sentences and everything after.

    Returns:
        (first_two, remainder) — either may be empty string if not enough content.
    """
    # Match sentence-ending punctuation (. ! ?) optionally followed by quotes/parens,
    # then whitespace. Handles abbreviations poorly by design (keep it simple).
    pattern = r'(?<=[.!?])["\')]?\s+'

    splits = list(re.finditer(pattern, text))

    if len(splits) == 0:
        # No sentence boundary found — everything is "first two"
        return text, ""
    elif len(splits) == 1:
        # Only one sentence boundary found
        boundary = splits[0].end()
        return text[:boundary].strip(), text[boundary:].strip()
    else:
        # Split after the 2nd sentence boundary
        boundary = splits[1].end()
        return text[:boundary].strip(), text[boundary:].strip()
    
def pretty(i, data): return "\n".join(f"{key}: {data[key][i]}" for key in data)