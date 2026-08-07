# must be run using conda env *llm_embeddings* on a GPU node!
#
# One-off: score the t=0 seed CSV (the un-attacked AI abstracts) through Fast-DetectGPT
# to create the baseline results_0_oss_{split}_{n}_fastdetect.csv that add_fastdetect.py
# reads for its original_* columns. Takes a t=0 CSV with an "original" column and emits
#   original, mirror_0, original_d, mirror_0_d, original_score_avg, mirror_0_score_avg
# where original == mirror_0 and the score columns are duplicated accordingly.
#
# Usage:
#   python make_t0_fastdetect.py results_0_oss_val_15.csv
#   python make_t0_fastdetect.py results_0_oss_test_50.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from add_fastdetect import score_abstracts, sigmoid

parser = argparse.ArgumentParser()
parser.add_argument("input_csv", help="t=0 CSV (must contain an 'original' column)")
parser.add_argument("--output_csv", default=None, help="default: <input>_fastdetect.csv")
args = parser.parse_args()

input_path = Path(args.input_csv)
output_path = Path(args.output_csv) if args.output_csv else input_path.with_name(
    input_path.stem + "_fastdetect.csv")

input_data = pd.read_csv(input_path)
assert "original" in input_data.columns, "'original' column not found in input CSV"

texts = input_data["original"].tolist()
d = score_abstracts(texts)
score_avg = sigmoid(d)

pd.DataFrame({
    "original": texts,
    "mirror_0": texts,
    "original_d": d,
    "mirror_0_d": d,
    "original_score_avg": score_avg,
    "mirror_0_score_avg": score_avg,
}).to_csv(output_path, index=False)

finite = d[np.isfinite(d)]
print(f"wrote {output_path}  (mean d={np.nanmean(d):.3f}, {len(finite)}/{len(d)} finite)")
