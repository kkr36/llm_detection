import re
import string
import time
from typing import Dict, List, Optional

import numpy as np
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
nlp.enable_pipe("senter")


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
            cleaned = abstract.lower().translate(punctuation_table)
            words = cleaned.split()
            num_words = len(words)

            if num_words == 0:
                continue

            total_words += num_words

            sentences = [
                s.strip() for s in sentence_splitter.split(abstract) if s.strip()
            ]
            total_sentences += len(sentences)

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
    except Exception:
        print(abstract, "bad")
        doc = nlp("")
    return [sent.text.strip() for sent in doc.sents]


def clean_text(text):
    bad_dashes = ['\u2014', '\u207b', '\u2013', '\u2011', '\u2010', '\u2212']
    bad_apostrophes = ['\u2019', '\u2032', '\u2018']
    bad_left_quote = '\u201c'
    bad_right_quote = '\u201d'
    for bad_dash in bad_dashes:
        text = [t.replace(bad_dash, '-') for t in text]
    for bad_apostrophe in bad_apostrophes:
        text = [t.replace(bad_apostrophe, "'") for t in text]
    text = [t.replace(bad_left_quote, '"') for t in text]
    text = [t.replace(bad_right_quote, '"') for t in text]

    text = [re.sub(r"[\n\t]+", " ", t) for t in text]
    allowed = set(string.printable)
    return [''.join(ch for ch in s if ch in allowed) for s in text]


def pretty(i, data):
    return "\n".join(f"{key}: {data[key][i]}" for key in data)


def predict_with_backoff(
    pangram_client,
    text: str,
    *,
    model: str,
    max_retries: int = 5,
    initial_delay: float = 1,
):
    """Call pangram_client.predict with model selection and exponential backoff."""
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return pangram_client.predict(text, model=model)
        except Exception as exc:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {exc}")
                return None

            print(f"Attempt {attempt + 1} failed: {exc}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

    return None


def _blank_pangram_score(model: str, error: str) -> dict:
    return {
        "pangram_model": model,
        "pangram_version": "",
        "pangram_prediction": "",
        "pangram_prediction_short": "",
        "fraction_ai": np.nan,
        "fraction_ai_assisted": np.nan,
        "fraction_human": np.nan,
        "num_ai_segments": np.nan,
        "window_labels": [],
        "window_ai_assistance_scores": [],
        "window_confidences": [],
        "window_is_humanized": [],
        "window_humanizer_scores": [],
        "pangram_error": error,
    }


def _check_version(
    result: dict,
    *,
    expected_version: Optional[str],
    expected_version_prefix: Optional[str],
):
    version = str(result.get("version", ""))
    if expected_version is not None and version != expected_version:
        raise RuntimeError(
            f"Expected Pangram version {expected_version}, got {version!r}."
        )
    if expected_version_prefix is not None and not version.startswith(expected_version_prefix):
        raise RuntimeError(
            f"Expected Pangram version prefix {expected_version_prefix!r}, got {version!r}."
        )


def flatten_pangram_result(
    result: dict,
    *,
    model: str,
    expected_version: Optional[str] = None,
    expected_version_prefix: Optional[str] = None,
) -> dict:
    _check_version(
        result,
        expected_version=expected_version,
        expected_version_prefix=expected_version_prefix,
    )

    windows = result.get("windows", []) or []
    return {
        "pangram_model": model,
        "pangram_version": result.get("version", ""),
        "pangram_prediction": result.get("prediction", ""),
        "pangram_prediction_short": result.get("prediction_short", ""),
        "fraction_ai": result.get("fraction_ai", np.nan),
        "fraction_ai_assisted": result.get("fraction_ai_assisted", np.nan),
        "fraction_human": result.get("fraction_human", np.nan),
        "num_ai_segments": result.get("num_ai_segments", np.nan),
        "window_labels": [window.get("label", None) for window in windows],
        "window_ai_assistance_scores": [
            window.get("ai_assistance_score", np.nan) for window in windows
        ],
        "window_confidences": [window.get("confidence", None) for window in windows],
        "window_is_humanized": [window.get("is_humanized", None) for window in windows],
        "window_humanizer_scores": [
            window.get("humanizer_score", np.nan) for window in windows
        ],
        "pangram_error": "",
    }


def score_pangram_text(
    pangram_client,
    text,
    *,
    model: str,
    expected_version: Optional[str] = None,
    expected_version_prefix: Optional[str] = None,
) -> dict:
    if not isinstance(text, str) or len(text) < 5:
        return _blank_pangram_score(model, "invalid text")
    if "sorry" in text.lower():
        return _blank_pangram_score(model, "possible model refusal")

    result = predict_with_backoff(pangram_client, text, model=model)
    if result is None or len(result) == 0:
        return _blank_pangram_score(model, "pangram request failed")

    return flatten_pangram_result(
        result,
        model=model,
        expected_version=expected_version,
        expected_version_prefix=expected_version_prefix,
    )


def collect_pangram_scores(
    pangram_client,
    texts,
    *,
    model: str,
    expected_version: Optional[str] = None,
    expected_version_prefix: Optional[str] = None,
    desc: Optional[str] = None,
) -> dict:
    rows = [
        score_pangram_text(
            pangram_client,
            text,
            model=model,
            expected_version=expected_version,
            expected_version_prefix=expected_version_prefix,
        )
        for text in tqdm(texts, desc=desc or f"scoring Pangram {model}")
    ]
    if not rows:
        return {}
    return {key: [row[key] for row in rows] for key in rows[0]}
