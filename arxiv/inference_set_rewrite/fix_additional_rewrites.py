import pandas as pd
from tqdm import tqdm

subsample_size = 20000 // 2
rewrite_pct = 0.2
category = "cs."
train_years = [str(x) for x in range(2010, 2021, 2)]
num_rewrite = int(subsample_size * rewrite_pct)  # 2000
gemini_mirror_idx = [i for i in range(num_rewrite) if i % 4 == 0]  # 500 indices

print(f"Processing years: {train_years}")
print(f"num_rewrite={num_rewrite}, gemini_mirror rows={len(gemini_mirror_idx)}")

for year in tqdm(train_years):
    v2_path = (
        f"/share/garg/arxiv_kaggle/multillm/double_rewrite/"
        f"arxiv_{year}_ai_{category}_{subsample_size}_{rewrite_pct}_fronthalf_120b_qwen_v2.parquet"
    )
    gemini_double_path = (
        f"/share/garg/arxiv_kaggle/multillm/double_rewrite/"
        f"arxiv_{year}_ai_{category}_{subsample_size}_{rewrite_pct}_fronthalf.parquet"
    )

    v2 = pd.read_parquet(v2_path)
    assert len(v2) == subsample_size + len(gemini_mirror_idx), (
        f"Year {year}: expected {subsample_size + len(gemini_mirror_idx)} rows, got {len(v2)}"
    )

    # Drop the incorrectly appended 500 rows — keep only the original subsample_size rows
    fixed = v2.iloc[:subsample_size].copy()

    # Load the double-mirror file and get the 500 Llama->Gemini3 rows
    gemini_double = pd.read_parquet(gemini_double_path)
    # gemini_double has the same 10000-row ordering as raw_data/fixed, so use
    # positional indexing directly — avoids false matches from duplicate abstracts
    double_mirror_values = gemini_double.iloc[gemini_mirror_idx]["Gemini 3 Preview"].values
    gem_col = fixed.columns.get_loc("Gemini 3 Preview")
    fixed.iloc[gemini_mirror_idx, gem_col] = double_mirror_values

    assert len(fixed) == subsample_size, (
        f"Year {year}: output has {len(fixed)} rows, expected {subsample_size}"
    )

    fixed.to_parquet(v2_path)
    print(f"Year {year}: fixed {len(gemini_mirror_idx)} Gemini3 double-mirror rows, saved to {v2_path}")
