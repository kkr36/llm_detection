"""Step 5: pass the 75 naive train mirrors (mirror_0) through Pangram 3.3.2.
This establishes how detectable the raw AI source is on the in-distribution CS set.
Cost: 75 Pangram units (~$0.05 each realtime).
"""
import json, re
import numpy as np, pandas as pd
from pangram import Pangram
from pangram_config import (PANGRAM_EXPECTED_VERSION, PANGRAM_EXPECTED_VERSION_PREFIX, PANGRAM_MODEL)
from util import collect_pangram_scores

IN_CSV = "results_0_oss_train_75.csv"
OUT_CSV = "results_0_oss_train_75_pangram.csv"

def rowpai(ws):
    vals = [v for v in (ws or []) if isinstance(v, (int, float)) and not pd.isna(v)]
    return float(np.mean(vals)) if vals else np.nan

if __name__ == "__main__":
    with open("/home/kkr36/creds.json") as h:
        client = Pangram(api_key=json.load(h)["pangram_api_key"])
    df = pd.read_csv(IN_CSV)
    scores = collect_pangram_scores(
        client, df["mirror_0"].tolist(), model=PANGRAM_MODEL,
        expected_version=PANGRAM_EXPECTED_VERSION, expected_version_prefix=PANGRAM_EXPECTED_VERSION_PREFIX)
    for k, v in scores.items():
        df[k] = v
    df["pai"] = df["window_ai_assistance_scores"].map(rowpai)
    df.to_csv(OUT_CSV, index=False)
    p = df["pai"]
    print(f"naive mirror_0 (t=0) on {len(df)} in-distribution CS train abstracts:")
    print(f"  mean P(AI)={p.mean():.3f}  median={p.median():.3f}")
    print(f"  labels: {df['pangram_prediction_short'].value_counts().to_dict()}")
    print(f"  fooled Human(<0.30): {(p<0.30).mean()*100:.1f}%   detected(>0.90): {(p>0.90).mean()*100:.1f}%")
    print(f"saved {OUT_CSV}")
