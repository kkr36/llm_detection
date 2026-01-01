import pandas as pd
import spacy
import string
import re
from tqdm import tqdm
import os
from collections import Counter

years = [2010, 2012, 2014, 2016, 2018, 2020]

for year in years:
    if not os.path.exists(str(year)):
        os.makedirs(str(year))

    path = f'/share/garg/arxiv_kaggle/multillm/double_rewrite/arxiv_{year}_ai_cs._10000_0.2_fronthalf.parquet'
    df = pd.read_parquet(path)

    # ---- CONFIG ----
    cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash", "human_abstract"]

    nlp = spacy.load("en_core_web_sm")
    allowed = set(string.printable)

    bad_dashes = ['—', '⁻', '–', '‑', '‐', '−']
    bad_apostrophes = ['’', "′", "‘"]
    bad_chars = bad_dashes + bad_apostrophes
    # import pdb; pdb.set_trace()

    # Create translation table for replacing bad chars
    trans_table = str.maketrans({ch: '' for ch in bad_chars})

    # =========================================================
    # Process each column ONCE and compute all metrics
    # =========================================================
    row_pct = {}
    sentence_pct = {}
    pct_rows_nonprintable = {}
    pct_sentences_nonprintable = {}
    pct_rows_nonprintable_clean = {}
    pct_sentences_nonprintable_clean = {}

    counts = {
        c: Counter("".join(df[c].astype(str)))
        for c in cols
    }

    # convert to frequency table
    char_freq = pd.DataFrame(counts).fillna(0).astype(int)

    for c in cols:
        print(f"\nProcessing column: {c}")
        series = df[c].dropna().astype(str)
        
        if len(series) == 0:
            continue

        # ---- ROW-LEVEL METRICS ----
        chars = set("".join(series))
        row_pct[c] = {
            ch: series.str.contains(re.escape(ch)).mean()
            for ch in chars
        }
        
        # Non-printable rows (original)
        mask = series.apply(lambda x: any(ch not in allowed for ch in x))
        pct_rows_nonprintable[c] = mask.mean()
        
        # Non-printable rows (after removing bad chars)
        series_clean = series.str.translate(trans_table)
        mask_clean = series_clean.apply(lambda x: any(ch not in allowed for ch in x))
        pct_rows_nonprintable_clean[c] = mask_clean.mean()
        
        # ---- SENTENCE-LEVEL METRICS (single pass) ----
        sentences = []
        for doc in tqdm(nlp.pipe(series, batch_size=100), total=len(series), desc=f"{c} sentences"):
            sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
        
        if sentences:
            sentences = pd.Series(sentences)
            sent_chars = set("".join(sentences))
            
            sentence_pct[c] = {
                ch: sentences.str.contains(re.escape(ch)).mean()
                for ch in sent_chars
            }
            
            # Non-printable sentences (original)
            mask = sentences.apply(lambda x: any(ch not in allowed for ch in x))
            pct_sentences_nonprintable[c] = mask.mean()
            
            # Non-printable sentences (after removing bad chars)
            sentences_clean = sentences.str.translate(trans_table)
            mask_clean = sentences_clean.apply(lambda x: any(ch not in allowed for ch in x))
            pct_sentences_nonprintable_clean[c] = mask_clean.mean()

    # =========================================================
    # Create DataFrames
    # =========================================================
    row_pct_df = pd.DataFrame(row_pct).fillna(0)
    sentence_pct_df = pd.DataFrame(sentence_pct).fillna(0)

    pct_rows_nonprintable_df = pd.Series(
        pct_rows_nonprintable, 
        name="pct_rows_non_printable"
    ).to_frame()

    pct_sentences_nonprintable_df = pd.Series(
        pct_sentences_nonprintable, 
        name="pct_sentences_non_printable"
    ).to_frame()

    pct_rows_nonprintable_clean_df = pd.Series(
        pct_rows_nonprintable_clean, 
        name="pct_rows_non_printable_clean"
    ).to_frame()

    pct_sentences_nonprintable_clean_df = pd.Series(
        pct_sentences_nonprintable_clean, 
        name="pct_sentences_non_printable_clean"
    ).to_frame()

    # Sort by human_abstract if present
    if "human_abstract" in row_pct_df.columns:
        row_pct_df = row_pct_df.sort_values(by="human_abstract", ascending=False)

    if "human_abstract" in sentence_pct_df.columns:
        sentence_pct_df = sentence_pct_df.sort_values(by="human_abstract", ascending=False)
    
    char_freq.to_csv(f"{year}/char_freq_df.csv")
    row_pct_df.to_csv(f"{year}/row_pct_df.csv")
    sentence_pct_df.to_csv(f"{year}/sentence_pct_df.csv")
    pct_rows_nonprintable_df.to_csv(f"{year}/pct_rows_nonprintable_df.csv")
    pct_sentences_nonprintable_df.to_csv(f"{year}/pct_sentences_nonprintable_df.csv")
    pct_rows_nonprintable_clean_df.to_csv(f"{year}/pct_rows_nonprintable_clean_df.csv")
    pct_sentences_nonprintable_clean_df.to_csv(f"{year}/pct_sentences_nonprintable_clean_df.csv")

    


# =========================================================
# --------------------- OUTPUTS ---------------------------
# =========================================================
# print("\n" + "="*60)
# print("% of ROWS containing each character (0–1):")
# print(row_pct_df.head(20))

# print("\n% of SENTENCES containing each character (0–1):")
# print(sentence_pct_df.head(20))

# print("\n% of ROWS containing ANY non-printable char (0–1):")
# print(pct_rows_nonprintable_df)

# print("\n% of SENTENCES containing ANY non-printable char (0–1):")
# print(pct_sentences_nonprintable_df)

# print("\n% of ROWS containing ANY non-printable char (cleaned, 0–1):")
# print(pct_rows_nonprintable_clean_df)

# print("\n% of SENTENCES containing ANY non-printable char (cleaned, 0–1):")
# print(pct_sentences_nonprintable_clean_df)

# import pdb; pdb.set_trace()