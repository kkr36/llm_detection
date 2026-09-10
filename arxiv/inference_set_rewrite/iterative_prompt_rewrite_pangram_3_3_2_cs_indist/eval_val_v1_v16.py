"""Evaluate v1 and v16 on the same 25-abstract VAL set, for comparison with v20."""
import json
import pandas as pd
from pangram import Pangram
from eval_val_v20_v21 import run_strategy, VAL_CSV

def main():
    val = pd.read_csv(VAL_CSV).reset_index(drop=True)
    with open("/home/kkr36/creds.json") as fh:
        pangram = Pangram(api_key=json.load(fh)["pangram_api_key"])
    for t in (1, 16):
        run_strategy(val, t, pangram)

if __name__ == "__main__":
    main()
