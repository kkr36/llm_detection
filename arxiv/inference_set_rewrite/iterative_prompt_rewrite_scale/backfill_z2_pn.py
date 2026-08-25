"""
Backfill empty rewrite_Z_2_PN rows only.
Re-runs the PN iteration-2 strategy (v33) on rows whose rewrite_Z_2_PN is empty/NaN
but whose rewrite_X source is valid, then writes the column back in place.
Same model + strategy as rewrite_x_mirror_z1.py.
"""
import sys
import concurrent.futures
import pandas as pd
from tqdm import tqdm

from openai_api import openai_oss_query
from strategy import z_mapping

model_name = "GPT OSS 120b"
method, iteration = "PN", 2
col = f"rewrite_Z_{iteration}_{method}"
source_col = "rewrite_X"
target_path = (
    "/share/garg/arxiv_kaggle/multillm/data_raw/"
    "arxiv_2020_xyz_v2_cs._10000_fronthalf.parquet"
)

strategy_fn = z_mapping[method][iteration]

if __name__ == "__main__":
    d = pd.read_parquet(target_path)
    assert col in d.columns and source_col in d.columns

    def is_missing(x):
        return (not isinstance(x, str)) or len(x.strip()) == 0

    def src_ok(x):
        return isinstance(x, str) and len(x.strip()) >= 10

    idx = [i for i in range(len(d))
           if is_missing(d[col].iloc[i]) and src_ok(d[source_col].iloc[i])]
    print(f"backfilling {len(idx)} rows of {col}")
    if not idx:
        print("nothing to do")
        sys.exit(0)

    col_vals = d[col].tolist()

    def process_row(i):
        new_text, _ = strategy_fn(openai_oss_query, d[source_col].iloc[i], model_name)
        return i, new_text

    filled = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(process_row, i): i for i in idx}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(idx),
                        desc=f"backfill {col}"):
            i, new_text = fut.result()
            if isinstance(new_text, str) and len(new_text.strip()) > 0:
                col_vals[i] = new_text
                filled += 1

    d[col] = col_vals
    d.to_parquet(target_path)
    still = sum(1 for i in range(len(d)) if is_missing(d[col].iloc[i]))
    print(f"filled {filled}/{len(idx)}; remaining empty {col}: {still}")
    print(f"saved column {col} into {target_path}")
