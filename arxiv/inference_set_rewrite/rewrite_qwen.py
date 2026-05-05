import concurrent.futures
import pandas as pd
from tqdm import tqdm
from qwen_api import qwen_query
from rewrite import rewrite_abstract

subsample_size = 20000
category = "cs."
years = [str(x) for x in range(2010, 2021, 2)][:]
gemini_col = "Gemini 2.5 Flash"
qwen_col = "Qwen"

for year in years:
    path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size//2}_fronthalf_120b.parquet"
    df = pd.read_parquet(path)

    mask = df[gemini_col].notna()
    indices = df.index[mask].tolist()
    abstracts = df.loc[indices, "human_abstract"].tolist()
    print(f"Year {year}: {len(indices)} rows to rewrite with Qwen")

    results = [None] * len(indices)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_pos = {
            executor.submit(rewrite_abstract, qwen_query, abstracts[i], qwen_col): i
            for i in range(len(indices))
        }
        for fut in tqdm(concurrent.futures.as_completed(future_to_pos), total=len(indices), desc=f"Year {year}"):
            pos = future_to_pos[fut]
            rewrite, model_name = fut.result()
            results[pos] = rewrite

    df[qwen_col] = None
    for pos, idx in enumerate(indices):
        df.at[idx, qwen_col] = results[pos]

    out_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size//2}_fronthalf_120b_qwen.parquet"
    df.to_parquet(out_path)
    print(f"Saved {out_path}")
