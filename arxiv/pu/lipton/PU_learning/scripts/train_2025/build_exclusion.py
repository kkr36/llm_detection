"""
Build the leakage-guard exclusion list for the 2025 TEDn training data.

Every abstract that appears anywhere in the downstream analysis
(pu/high_conf_human_analysis_cs) must be kept out of training.  The canonical
superset of that analysis is the two `sampled_abstracts_{2020,2025}.csv` files,
which carry the raw abstract text (all other CSVs in that tree are sentence-level
subsets of these, keyed by abstract_id).  We normalize each abstract and write
one key per line to downstream_exclude_norm.txt, which read_2025.py loads.

Matching is whitespace/punctuation-insensitive and lowercased, so formatting
differences between the parquet and the CSVs don't matter.

  python scripts/train_2025/build_exclusion.py
"""

import csv
import os
import re

DOWNSTREAM_DIR = "/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis_cs/data"
SAMPLED_FILES = ["sampled_abstracts_2020.csv", "sampled_abstracts_2025.csv"]
OUT_PATH = os.path.join(os.path.dirname(__file__), "downstream_exclude_norm.txt")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower()) if isinstance(s, str) else ""


def main():
    excl = set()
    for fname in SAMPLED_FILES:
        path = os.path.join(DOWNSTREAM_DIR, fname)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            assert "abstract" in reader.fieldnames, f"{fname}: no 'abstract' column ({reader.fieldnames})"
            n = 0
            for row in reader:
                key = norm(row["abstract"])
                if key:
                    excl.add(key)
                    n += 1
        print(f"{fname}: {n} abstracts")

    with open(OUT_PATH, "w") as fh:
        for key in sorted(excl):
            fh.write(key + "\n")
    print(f"wrote {len(excl)} unique normalized abstracts -> {OUT_PATH}")


if __name__ == "__main__":
    main()
