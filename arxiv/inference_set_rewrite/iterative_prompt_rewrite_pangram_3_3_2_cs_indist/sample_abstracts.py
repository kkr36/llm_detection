"""Steps 2-3: sample 100 in-distribution 2020 CS abstracts, disjoint from the multillm 20k,
then split 75/25 train/val.

Replicates the EXACT multillm sampling filter (james_analysis/subsample_by_year.py):
    - category filter:  "cs." in entry["categories"]   (substring; also catches physics.* by
      design of the original pipeline -> keeps us in-distribution to the eval)
    - year:             entry["update_date"].split("-")[0] == "2020"
Mutual exclusivity with the multillm 20k is enforced by exact-abstract-string set difference
against arxiv-metadata-oai-snapshot_cs._20000.json["2020"].
"""
import json, random
import pandas as pd

SNAPSHOT = "/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json"
MULTILLM_20K = "/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_cs._20000.json"
CATEGORY = "cs."
YEAR = "2020"
N_SAMPLE = 100
N_TRAIN = 75
SEED = 2024

# exclusion set: the exact 20k 2020 abstracts already used by multillm
excl = set(json.load(open(MULTILLM_20K))[YEAR])
print(f"multillm 2020 exclusion set: {len(excl)}")

# stream the full snapshot, replicate the multillm filter
pool = []  # (abstract, id, categories)
seen_abs = set()
n_lines = n_cs2020 = 0
with open(SNAPSHOT, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        n_lines += 1
        e = json.loads(line)
        cats = e.get("categories", "") or ""
        if CATEGORY not in cats:
            continue
        ud = e.get("update_date")
        if not ud or ud.split("-")[0] != YEAR:
            continue
        n_cs2020 += 1
        ab = e.get("abstract")
        if not ab:
            continue
        if ab in excl:            # disjoint from multillm 20k
            continue
        if ab in seen_abs:        # dedupe within pool
            continue
        seen_abs.add(ab)
        pool.append((ab, e.get("id"), cats))

print(f"scanned {n_lines} records; cs.-2020 matches {n_cs2020}; "
      f"held-out (not in 20k, deduped) pool = {len(pool)}")
assert len(pool) >= N_SAMPLE, "not enough held-out abstracts"

rng = random.Random(SEED)
sample = rng.sample(pool, N_SAMPLE)
rng.shuffle(sample)
train = sample[:N_TRAIN]
val = sample[N_TRAIN:]

def to_df(rows):
    return pd.DataFrame({
        "arxiv_id": [r[1] for r in rows],
        "categories": [r[2] for r in rows],
        "human_abstract": [r[0] for r in rows],
    })

to_df(train).to_csv("abstracts_train_75.csv", index=False)
to_df(val).to_csv("abstracts_val_25.csv", index=False)

# sanity: disjoint from exclusion set
assert not (set(r[0] for r in sample) & excl), "overlap with multillm 20k!"
print(f"saved abstracts_train_75.csv ({len(train)}) and abstracts_val_25.csv ({len(val)})")
print(f"disjoint from multillm 20k: OK (seed={SEED})")
# quick category peek
cat_counts = {}
for r in sample:
    prim = r[2].split()[0] if r[2] else "?"
    cat_counts[prim] = cat_counts.get(prim, 0) + 1
print("primary-category counts (top):",
      dict(sorted(cat_counts.items(), key=lambda x: -x[1])[:12]))
