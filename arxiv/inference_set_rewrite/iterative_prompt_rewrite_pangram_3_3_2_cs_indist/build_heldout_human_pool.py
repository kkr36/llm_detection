"""Build a held-out human-abstract pool for few-shot style emulation (t>=7).

Guarantees required by the task:
- No pool abstract is (or begins with) any of the 75 train `original` abstracts,
  enforced by normalized full-text AND 150-char-prefix exclusion. So for every
  train row, the few-shot human examples cannot contain that row's original.
- Distribution match to the train pool: primary category drawn from the set of
  train primary categories, and the 2019-2020 arxiv era (YYMM 1901..2012).

Source: /share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json (one JSON/line,
sorted by id ascending, so we can stop once we pass the era window).

Output: heldout_human_abstracts.json -- a list of {id, categories, abstract}.
Deterministic (reservoir seed fixed). Run in the llm_master env.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
KAGGLE = Path("/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json")
OUT = HERE / "heldout_human_abstracts.json"
TRAIN = HERE / "results_0_oss_train_75.csv"

POOL_SIZE = 800
MIN_WORDS, MAX_WORDS = 60, 300
ERA_LO, ERA_HI = 1901, 2012  # 2019-01 .. 2020-12, matching the train ids
RESERVOIR_SEED = 20240
_norm_re = re.compile(r"[^a-z0-9]")


def norm(text):
    return _norm_re.sub("", str(text).lower())


def yymm(arxiv_id):
    """Leading YYMM integer of a modern arxiv id like '2007.12345'."""
    head = str(arxiv_id).split(".")[0]
    return int(head[:4]) if head[:4].isdigit() else None


def main():
    train = pd.read_csv(TRAIN)
    train_full = {norm(o) for o in train["original"]}
    train_prefix = {norm(o)[:150] for o in train["original"]}
    train_primary = {c.split()[0] for c in train["categories"]}
    print(f"train originals to exclude: {len(train_full)}")
    print(f"train primary categories: {sorted(train_primary)}")

    rng = random.Random(RESERVOIR_SEED)
    reservoir = []
    seen = 0
    scanned = 0
    started_era = False

    with open(KAGGLE, "r") as fh:
        for line in fh:
            scanned += 1
            if scanned % 500000 == 0:
                print(f"  scanned {scanned:,} lines, kept-candidates seen {seen:,}")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ym = yymm(rec.get("id", ""))
            if ym is None:
                continue
            if ym < ERA_LO:
                continue
            if ym > ERA_HI:
                # ids are globally sorted; once past the era we are done.
                break
            started_era = True

            primary = (rec.get("categories") or "").split()
            if not primary or primary[0] not in train_primary:
                continue
            abstract = (rec.get("abstract") or "").strip()
            abstract = re.sub(r"\s+", " ", abstract)
            n = len(abstract.split())
            if n < MIN_WORDS or n > MAX_WORDS:
                continue
            na = norm(abstract)
            if na in train_full or na[:150] in train_prefix:
                continue  # never leak a train original

            seen += 1
            item = {
                "id": rec["id"],
                "categories": rec["categories"],
                "abstract": abstract,
            }
            # reservoir sampling to POOL_SIZE
            if len(reservoir) < POOL_SIZE:
                reservoir.append(item)
            else:
                j = rng.randint(0, seen - 1)
                if j < POOL_SIZE:
                    reservoir[j] = item

    print(f"done. scanned {scanned:,} lines, era-matching candidates {seen:,}")
    print(f"pool size: {len(reservoir)}")

    # final safety re-check: no pool abstract collides with a train original
    collisions = 0
    for it in reservoir:
        na = norm(it["abstract"])
        if na in train_full or na[:150] in train_prefix:
            collisions += 1
    print(f"post-hoc collisions with train originals: {collisions}")
    assert collisions == 0

    from collections import Counter
    cats = Counter(it["categories"].split()[0] for it in reservoir)
    print("pool primary-category spread (top 15):", cats.most_common(15))

    OUT.write_text(json.dumps(reservoir, indent=0))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
