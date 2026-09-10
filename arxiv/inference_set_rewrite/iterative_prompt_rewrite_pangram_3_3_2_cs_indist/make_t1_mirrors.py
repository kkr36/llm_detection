"""Step 6 (t=1): apply the reference prompt (rewrite_function_1) to a random sample of 25 of the
naive train mirrors (mirror_0 -> mirror_1, i.e. humanize the AI source), then Pangram-score
those 25. Compares directly against their t=0 (naive) scores from step 5.
Cost: 25 Pangram units (~$0.05 each).
"""
import json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
from tqdm import tqdm
from pangram import Pangram

from openai_api import openai_oss_query
from strategy import rewrite_function_1
from pangram_config import (PANGRAM_EXPECTED_VERSION, PANGRAM_EXPECTED_VERSION_PREFIX, PANGRAM_MODEL)
from util import collect_pangram_scores

LLM_LABEL = "GPT OSS 120b"
N_T1 = 25
SEED = 2024

def rowpai(ws):
    vals = [v for v in (ws or []) if isinstance(v, (int, float)) and not pd.isna(v)]
    return float(np.mean(vals)) if vals else np.nan

if __name__ == "__main__":
    train = pd.read_csv("results_0_oss_train_75.csv")
    sub = train.sample(n=N_T1, random_state=SEED).reset_index(drop=True)
    sources = sub["mirror_0"].tolist()

    mirrors = [None] * N_T1
    def work(i):
        m, _ = rewrite_function_1(openai_oss_query, sources[i], LLM_LABEL)
        return i, m
    print(f"applying rewrite_function_1 to {N_T1} naive mirrors (t=0 -> t=1) ...")
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(work, i) for i in range(N_T1)]
        for fut in tqdm(as_completed(futs), total=N_T1):
            i, m = fut.result()
            mirrors[i] = m
    sub["mirror_1"] = mirrors

    with open("/home/kkr36/creds.json") as h:
        client = Pangram(api_key=json.load(h)["pangram_api_key"])
    print("scoring 25 t=1 mirrors with Pangram 3.3.2 ...")
    scores = collect_pangram_scores(
        client, mirrors, model=PANGRAM_MODEL,
        expected_version=PANGRAM_EXPECTED_VERSION, expected_version_prefix=PANGRAM_EXPECTED_VERSION_PREFIX)
    for k, v in scores.items():
        sub[k] = v
    sub["pai_t1"] = sub["window_ai_assistance_scores"].map(rowpai)
    sub.to_csv("results_1_oss_train_25_pangram.csv", index=False)

    p = sub["pai_t1"]
    print(f"\nt=1 (rewrite_function_1) on {N_T1} in-distribution CS abstracts:")
    print(f"  mean P(AI)={p.mean():.3f}  median={p.median():.3f}")
    print(f"  labels: {sub['pangram_prediction_short'].value_counts().to_dict()}")
    print(f"  fooled Human(<0.30): {(p<0.30).mean()*100:.1f}%   detected(>0.90): {(p>0.90).mean()*100:.1f}%")
    print("saved results_1_oss_train_25_pangram.csv")
