import pandas as pd
import spacy
import string
import re
from tqdm import tqdm

year = 2020
path = f'/share/garg/arxiv_kaggle/multillm/double_rewrite/arxiv_{year}_ai_cs._10000_0.2_fronthalf.parquet'
df = pd.read_parquet(path)

# ---- CONFIG ----
cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash", "human_abstract"]   # string columns you want to analyze

# -------- CONFIG --------
nlp = spacy.load("en_core_web_sm")
allowed = set(string.printable)

# =========================================================
# 1️⃣ % of ROWS containing each character (per column)
# =========================================================
row_pct = {}

for c in cols:
    series = df[c].dropna().astype(str)
    if len(series) == 0:
        continue

    chars = set("".join(series))
    row_pct[c] = {
        ch: series.str.contains(re.escape(ch)).mean()
        for ch in chars
    }

row_pct_df = pd.DataFrame(row_pct).fillna(0)


# =========================================================
# 2️⃣ % of SENTENCES containing each character (per column, spaCy)
# =========================================================
sentence_pct = {}

for c in cols:
    series = df[c].dropna().astype(str)

    sentences = []
    for doc in tqdm(nlp.pipe(series, batch_size=100), total=len(series) // 100):
        sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])

    if not sentences:
        continue

    sentences = pd.Series(sentences)
    chars = set("".join(sentences))

    sentence_pct[c] = {
        ch: sentences.str.contains(re.escape(ch)).mean()
        for ch in chars
    }

sentence_pct_df = pd.DataFrame(sentence_pct).fillna(0)


# =========================================================
# 3️⃣ % of ROWS containing ANY non-printable (non-keyboard) character
# =========================================================
pct_rows_nonprintable = {}

for c in cols:
    s = df[c].dropna().astype(str)

    mask = s.apply(lambda x: any(ch not in allowed for ch in x))
    pct_rows_nonprintable[c] = mask.mean()

pct_rows_nonprintable = pd.Series(pct_rows_nonprintable, name="pct_rows_non_printable").to_frame()


# =========================================================
# 4️⃣ % of SENTENCES containing ANY non-printable character
# =========================================================
pct_sentences_nonprintable = {}

for c in cols:
    series = df[c].dropna().astype(str)

    sentences = []
    for doc in tqdm(nlp.pipe(series, batch_size=100), total=len(series) // 100):
        sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])

    if not sentences:
        continue

    sentences = pd.Series(sentences)

    mask = sentences.apply(lambda x: any(ch not in allowed for ch in x))
    pct_sentences_nonprintable[c] = mask.mean()

pct_sentences_nonprintable = pd.Series(
    pct_sentences_nonprintable, 
    name="pct_sentences_non_printable"
).to_frame()

bad_dashes = ['‑', '–', '⁻', '−']
bad_apostrophes = ['’', '′', '‘']
bad_chars = bad_dashes + bad_apostrophes

# =========================================================
# 3️⃣ % of ROWS containing ANY non-printable (non-keyboard) character, after replacing dashes/apostrophes
# =========================================================
pct_rows_nonprintable = {}

for c in cols:
    s = df[c].dropna().astype(str)
    s = s.apply(lambda x: any(ch not in bad_chars for ch in x))

    mask = s.apply(lambda x: any(ch not in allowed for ch in x))
    pct_rows_nonprintable[c] = mask.mean()

pct_rows_nonprintable = pd.Series(pct_rows_nonprintable, name="pct_rows_non_printable").to_frame()


# =========================================================
# 4️⃣ % of SENTENCES containing ANY non-printable character, after replacing dashes/apostrophes
# =========================================================
pct_sentences_nonprintable = {}

for c in cols:
    series = df[c].dropna().astype(str)
    series = series.apply(lambda x: any(ch not in bad_chars for ch in x))

    sentences = []
    for doc in tqdm(nlp.pipe(series, batch_size=100), total=len(series) // 100):
        sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])

    if not sentences:
        continue

    sentences = pd.Series(sentences)

    mask = sentences.apply(lambda x: any(ch not in allowed for ch in x))
    pct_sentences_nonprintable[c] = mask.mean()

pct_sentences_nonprintable = pd.Series(
    pct_sentences_nonprintable, 
    name="pct_sentences_non_printable"
).to_frame()

if "human_abstract" in row_pct_df.columns:
    row_pct_df = row_pct_df.sort_values(by="human_abstract", ascending=False)

if "human_abstract" in sentence_pct_df.columns:
    sentence_pct_df = sentence_pct_df.sort_values(by="human_abstract", ascending=False)


# =========================================================
# --------------------- OUTPUTS ---------------------------
# =========================================================
print("\n% of ROWS containing each character (0–1):")
print(row_pct_df)

print("\n% of SENTENCES containing each character (0–1):")
print(sentence_pct_df)

print("\n% of ROWS containing ANY non-printable char (0–1):")
print(pct_rows_nonprintable)

print("\n% of SENTENCES containing ANY non-printable char (0–1):")
print(pct_sentences_nonprintable)

import pdb; pdb.set_trace()




# # -------- CONFIG --------
# nlp = spacy.load("en_core_web_sm")

# # -------- 1️⃣ PERCENT OF ROWS PER COLUMN CONTAINING EACH CHARACTER --------
# row_pct = {}

# for c in cols:
#     series = df[c].dropna().astype(str)

#     if len(series) == 0:
#         continue

#     chars = set("".join(series))

#     row_pct[c] = {
#         ch: series.str.contains(re.escape(ch)).mean()
#         for ch in chars
#     }

# row_pct_df = pd.DataFrame(row_pct).fillna(0)


# # -------- 2️⃣ PERCENT OF SENTENCES PER COLUMN CONTAINING EACH CHARACTER (spaCy) --------
# sentence_pct = {}

# for c in cols:
#     series = df[c].dropna().astype(str)

#     # sentence segmentation w/ spaCy
#     sentences = []
#     for doc in nlp.pipe(series, batch_size=100):
#         sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])

#     if not sentences:
#         continue

#     sentences = pd.Series(sentences)
#     chars = set("".join(sentences))

#     sentence_pct[c] = {
#         ch: sentences.str.contains(re.escape(ch)).mean()
#         for ch in chars
#     }

# sentence_pct_df = pd.DataFrame(sentence_pct).fillna(0)

# # ---- CHARACTER COUNTS (ignores nulls) ----
# counts = {
#     c: Counter("".join(df[c].dropna().astype(str)))
#     for c in cols
# }

# # Frequency table: rows = characters, columns = dataframe columns
# char_freq = pd.DataFrame(counts).fillna(0).astype(int)

# # ---- NORMALIZED FREQUENCIES (per column, comparable) ----
# char_freq_norm = char_freq.div(char_freq.sum(axis=0), axis=1)

