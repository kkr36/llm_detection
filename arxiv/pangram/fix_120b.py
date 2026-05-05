"""
Fix mislabeled GPT OSS data:
  - all_100_2010.csv: rename "GPT OSS 120b" col -> "GPT OSS 20b", generate real
    120b rewrites, save as "GPT OSS 120b"
  - results_everything_100_2010.csv: rename source "GPT OSS 120b" -> "GPT OSS 20b",
    run new 120b rewrites through pangram, append 100 new rows with source "GPT OSS 120b"
"""

from pangram import Pangram
import json
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
import concurrent.futures
from oss_api import openai_oss_120b_query

# ── credentials ──────────────────────────────────────────────────────────────
with open("/home/kkr36/creds.json", "r") as f:
    creds = json.load(f)
pangram_client = Pangram(api_key=creds["pangram_api_key"])

DATA_CSV    = "all_100_2010.csv"
RESULTS_CSV = "results_everything_100_2010.csv"  # "results_all_100_2010.csv" in task description

# ── rewrite helpers (same 3-step chain as write_all.py) ──────────────────────
def rewrite_abstract(abstract):
    context1 = (
        "The aim here is to reverse-engineer the author's writing process by "
        "compressing a piece of text into a concise structured form. "
        "First summarize the goal of the text (introduction, method, results, etc.) "
        "and then reverse-engineer it into bullet points."
    )
    res1 = openai_oss_120b_query(context1, f"Here is the text: {abstract}")

    context2 = "Expand the concise bullet-point version into a full abstract."
    res2 = openai_oss_120b_query(context2, f"Here is the writing: {res1}")

    context3 = (
        "Proofread the writing for grammatical accuracy. "
        "Return ONLY the corrected abstract with no titles or extra text."
    )
    res3 = openai_oss_120b_query(context3, f"Here is the writing: {res2}")
    return res3


# ── pangram helper ────────────────────────────────────────────────────────────
def predict_with_backoff(text, max_retries=5, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return pangram_client.predict(text)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 & 2 & 3 — rewrite CSVs
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading", DATA_CSV)
data = pd.read_csv(DATA_CSV)

# rename mislabeled column
data = data.rename(columns={"GPT OSS 120b": "GPT OSS 20b"})

abstracts = data["human_abstract"].tolist()
assert len(abstracts) == 100, f"Expected 100 abstracts, got {len(abstracts)}"

print("Rewriting 100 abstracts with GPT OSS 120b (openai_oss_120b_query)...")
rewrites = [None] * len(abstracts)

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    future_to_idx = {executor.submit(rewrite_abstract, ab): i for i, ab in enumerate(abstracts)}
    for fut in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(abstracts)):
        idx = future_to_idx[fut]
        try:
            rewrites[idx] = fut.result()
        except Exception as e:
            print(f"Rewrite failed for idx={idx}: {e}")

data["GPT OSS 120b"] = rewrites
data.to_csv(DATA_CSV.replace(".csv", "_2.csv"), index=False)
print(f"Saved updated {DATA_CSV.replace('.csv', '_2.csv')}  (GPT OSS 20b + GPT OSS 120b columns)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 & 5 — update pangram results
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading", RESULTS_CSV)
results_df = pd.read_csv(RESULTS_CSV)

# rename mislabeled source rows
results_df["source"] = results_df["source"].replace("GPT OSS 120b", "GPT OSS 20b")

print("Running 100 new GPT OSS 120b rewrites through pangram API...")
new_rows = []
failed = []

def pangram_for_idx(idx, text):
    if text is None:
        print(f"Skipping idx={idx} (rewrite failed)")
        return idx, None
    result = predict_with_backoff(text)
    if result is None or len(result) == 0:
        print(f"Skipping pangram for idx={idx}")
        return idx, None
    window_labels, window_ai_scores, window_confs = [], [], []
    for w in result.get("windows", []):
        window_labels.append(w.get("label", None))
        window_ai_scores.append(w.get("ai_assistance_score", np.nan))
        window_confs.append(w.get("confidence", np.nan))
    return idx, {
        "year": 2010,
        "idx": idx,
        "source": "GPT OSS 120b",
        "fraction_ai": result.get("fraction_ai", np.nan),
        "fraction_ai_assisted": result.get("fraction_ai_assisted", np.nan),
        "fraction_human": result.get("fraction_human", np.nan),
        "num_ai_segments": result.get("num_ai_segments", np.nan),
        "window_labels": window_labels,
        "window_ai_assistance_scores": window_ai_scores,
        "window_confidences": window_confs,
        "text": text,
    }

with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
    future_to_idx = {executor.submit(pangram_for_idx, i, text): i for i, text in enumerate(rewrites)}
    for fut in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(rewrites), desc="pangram-120b"):
        try:
            idx, row = fut.result()
            if row is None:
                failed.append(idx)
            else:
                new_rows.append(row)
        except Exception as e:
            idx = future_to_idx[fut]
            print(f"Failed to process pangram result for idx={idx}: {e}")
            failed.append(idx)

new_df = pd.DataFrame(new_rows)
results_df = pd.concat([results_df, new_df], ignore_index=True)
results_df.to_csv(RESULTS_CSV.replace(".csv", "_2.csv"), index=False)
print(f"Saved updated {RESULTS_CSV.replace('.csv', '_2.csv')}  ({len(results_df)} rows, "
      f"{len(new_rows)} new GPT OSS 120b rows)")

if failed:
    print(f"\nFailed indices: {failed}")
