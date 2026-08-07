"""
Join arXiv subject categories onto the sampled abstracts by streaming the full
metadata snapshot once. The parquet `human_abstract` is the front-half of the
official abstract, so a whitespace-normalized 100-char prefix is a stable join key.

Writes ../data/categories.parquet: prefix_key -> categories, primary_category.
"""
import json
import pandas as pd

META = "/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json"
OUT = "/home/kkr36/llm_detection/arxiv/temporal_human_analysis/data"


def prefix_key(text, n=100):
    return " ".join(str(text).split())[:n]


# collect all keys we need (both years, abstract + pseudolabel files)
keys = set()
for f in ["2020_human_sample.csv", "2025_abstract_predictions.csv"]:
    df = pd.read_csv(f"{OUT}/{f}")
    keys.update(df["prefix_key"].tolist())
print(f"[categories] need {len(keys)} distinct abstract keys")

found = {}
n_lines = 0
with open(META, "r") as fh:
    for line in fh:
        n_lines += 1
        if n_lines % 500000 == 0:
            print(f"[categories] scanned {n_lines} records, matched {len(found)}/{len(keys)}")
        if len(found) == len(keys):
            break
        # cheap pre-filter to skip json.loads on obvious non-matches is hard here;
        # abstracts are large, so just parse.
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ab = rec.get("abstract")
        if not ab:
            continue
        k = prefix_key(ab)
        if k in keys and k not in found:
            cats = rec.get("categories", "") or ""
            found[k] = {
                "prefix_key": k,
                "categories": cats,
                "primary_category": cats.split()[0] if cats.split() else "",
                "arxiv_id": rec.get("id", ""),
                "update_date": rec.get("update_date", ""),
            }

print(f"[categories] matched {len(found)}/{len(keys)} keys after {n_lines} records")
cat_df = pd.DataFrame(list(found.values()))
cat_df.to_parquet(f"{OUT}/categories.parquet", index=False)
print("[categories] wrote categories.parquet")
