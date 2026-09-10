### Pangram-only rescoring helper for an existing generated-results CSV.
# No local detector models are loaded or used here.
# must be run using conda env *llm_master*!

import argparse
import json
from pathlib import Path

import pandas as pd
from pangram import Pangram

from pangram_config import (
    PANGRAM_EXPECTED_VERSION,
    PANGRAM_EXPECTED_VERSION_PREFIX,
    PANGRAM_MODEL,
)
from strategy import CURRENT_TIMESTEP
from util import collect_pangram_scores

with open("/home/kkr36/creds.json", "r") as handle:
    pangram_api_key = json.load(handle)["pangram_api_key"]
pangram_client = Pangram(api_key=pangram_api_key)

parser = argparse.ArgumentParser()
parser.add_argument("--timestep", default=str(CURRENT_TIMESTEP))
parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--to-mirror", type=int, default=15)
parser.add_argument("--input-csv", default=None)
parser.add_argument("--output-csv", default=None)
parser.add_argument(
    "--mirror-column",
    default=None,
    help="Mirror column to score; defaults to mirror_{timestep}.",
)
args = parser.parse_args()

input_csv = args.input_csv or f"results_{args.timestep}_oss_{args.split}_{args.to_mirror}.csv"
output_csv = args.output_csv or input_csv.replace(".csv", "_pangram.csv")
mirror_col = args.mirror_column or f"mirror_{args.timestep}"

if __name__ == "__main__":
    if Path(output_csv).exists():
        raise FileExistsError(f"refusing to overwrite {output_csv}")
    input_data = pd.read_csv(input_csv)
    if mirror_col not in input_data.columns:
        raise KeyError(f"{mirror_col!r} not found in {input_csv}")

    pangram_scores = collect_pangram_scores(
        pangram_client,
        input_data[mirror_col].tolist(),
        model=PANGRAM_MODEL,
        expected_version=PANGRAM_EXPECTED_VERSION,
        expected_version_prefix=PANGRAM_EXPECTED_VERSION_PREFIX,
    )
    for key, values in pangram_scores.items():
        input_data[key] = values

    input_data.to_csv(output_csv, index=False)
    print(f"saved {output_csv}")
