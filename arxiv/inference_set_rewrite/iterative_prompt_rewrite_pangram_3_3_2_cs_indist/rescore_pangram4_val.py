"""Re-score the EXISTING v16 and v20 val rewrites with Pangram 4 (no regeneration).
Same mirror_16 / mirror_20 text that Pangram 3.3.2 already scored -> isolates the
detector-version effect. Writes results_{16,20}_oss_val_25_pangram4.csv."""
import json
import numpy as np
import pandas as pd
from pangram import Pangram
from util import collect_pangram_scores

def row_pai(ws):
    vals = [s for s in (ws or []) if isinstance(s, (int, float)) and not pd.isna(s)]
    return float(np.mean(vals)) if vals else np.nan

def main():
    with open("/home/kkr36/creds.json") as fh:
        pangram = Pangram(api_key=json.load(fh)["pangram_api_key"])
    for t in (16, 20):
        df = pd.read_csv(f"results_{t}_oss_val_25_pangram.csv")
        texts = df[f"mirror_{t}"].tolist()
        scores = collect_pangram_scores(
            pangram, texts, model="pangram-4",
            expected_version=None, expected_version_prefix=None,
            desc=f"pangram-4 v{t}",
        )
        for k, v in scores.items():
            df[f"p4_{k}"] = v
        df[f"p4_pai_t{t}"] = df["p4_window_ai_assistance_scores"].map(row_pai)
        dest = f"results_{t}_oss_val_25_pangram4.csv"
        df.to_csv(dest, index=False)
        ver = pd.Series(df["p4_pangram_version"]).unique().tolist()
        print(f"\nv{t}: pangram_version(s)={ver}")
        print("  p4 prediction_short:", df["p4_pangram_prediction_short"].value_counts().to_dict())
        print(f"  p4 mean fraction_ai={df['p4_fraction_ai'].mean():.3f}  mean pai={df[f'p4_pai_t{t}'].mean():.3f}")
        print(f"  saved {dest}")

if __name__ == "__main__":
    main()
