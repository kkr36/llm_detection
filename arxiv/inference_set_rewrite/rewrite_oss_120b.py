import concurrent.futures
import pandas as pd
from tqdm import tqdm
from openai_api import openai_oss_120b_query
from rewrite import rewrite_abstract

subsample_size = 20000
category = "cs."
years = [str(x) for x in range(2010, 2021, 2)][-1:]
orig_col = "GPT OSS 120b"
renamed_col = "GPT OSS 20b"

for year in years:
    path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size//2}_fronthalf.parquet"
    df = pd.read_parquet(path)

    df = df.rename(columns={orig_col: renamed_col})

    mask = df[renamed_col].notna()
    indices = df.index[mask].tolist()
    abstracts = df.loc[indices, "human_abstract"].tolist()
    print(f"Year {year}: {len(indices)} rows to rewrite with 120b")

    results = [None] * len(indices)

    import pdb; pdb.set_trace()

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_pos = {
            executor.submit(rewrite_abstract, openai_oss_120b_query, abstracts[i], orig_col): i
            for i in range(len(indices))
        }
        for fut in tqdm(concurrent.futures.as_completed(future_to_pos), total=len(indices), desc=f"Year {year}"):
            pos = future_to_pos[fut]
            rewrite, model_name = fut.result()
            results[pos] = rewrite

    df[orig_col] = None
    for pos, idx in enumerate(indices):
        df.at[idx, orig_col] = results[pos]

    out_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size//2}_fronthalf_120b.parquet"
    df.to_parquet(out_path)
    print(f"Saved {out_path}")
