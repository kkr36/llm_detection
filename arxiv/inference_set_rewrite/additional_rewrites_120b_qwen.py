import os
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from llama_api import llama_query
from openai_api import openai_oss_120b_query
from qwen_api import qwen_query
from rewrite import rewrite_abstract

# Double-mirror chain: Llama -> Gemini 3 -> GPT OSS 120b -> Qwen -> Llama
# Gemini 3 mirrors of Llama are read from existing files; only the remaining
# three links are computed here.
llm_labels = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Qwen"]
query_fns = [llama_query, None, openai_oss_120b_query, qwen_query]
assert len(query_fns) == len(llm_labels)

if __name__ == "__main__":
    subsample_size = 20000 // 2
    rewrite_pct = 0.2
    category = "cs."
    train_years = [str(x) for x in range(2010, 2021, 2)]
    print(train_years)

    for year in tqdm(train_years):

        # Step 1: Read raw data (has single mirrors for all 4 LLMs + 2 garbage cols)
        raw_path = (
            f"/share/garg/arxiv_kaggle/multillm/data_raw/"
            f"arxiv_{year}_ai_{category}_{subsample_size}_fronthalf_120b_qwen.parquet"
        )
        raw_data = pd.read_parquet(raw_path)
        assert len(raw_data) == subsample_size

        keep_cols = ["human_abstract"] + llm_labels
        arxiv_data = raw_data[keep_cols].copy()
        num_rewrite = int(len(arxiv_data) * rewrite_pct)

        # Step 2: Read existing double_rewrite file which has Gemini 3 mirrors of Llama
        gemini_double_path = (
            f"/share/garg/arxiv_kaggle/multillm/double_rewrite/"
            f"arxiv_{year}_ai_{category}_{subsample_size}_{rewrite_pct}_fronthalf.parquet"
        )
        gemini_double_data = pd.read_parquet(gemini_double_path)

        # Step 3: Extract the 500 rows where Gemini 3 mirrors Llama (i % 4 == 0)
        # and append them as new rows to arxiv_data
        gemini_mirror_idx = [i for i in range(num_rewrite) if i % 4 == 0]
        extra_rows = gemini_double_data.iloc[gemini_mirror_idx][
            ["human_abstract", "Llama 3.3 70b Instruct", "Gemini 3 Preview"]
        ].copy()
        extra_rows["GPT OSS 120b"] = pd.NA
        extra_rows["Qwen"] = pd.NA
        extra_rows = extra_rows[keep_cols].reset_index(drop=True)

        combined = pd.concat([arxiv_data, extra_rows], ignore_index=True)
        assert len(combined) == len(arxiv_data) + len(extra_rows), (
            f"Expected {len(arxiv_data) + len(extra_rows)} rows, got {len(combined)}"
        )
        print(
            f"Year {year}: confirmed {len(arxiv_data)} original + {len(extra_rows)} "
            f"Llama->Gemini3 rows = {len(combined)} total (no rows removed)"
        )

        # Step 4: Compute the remaining double mirrors on the first len(arxiv_data) rows
        # i % 4 == 1: Gemini 3 -> GPT OSS 120b
        # i % 4 == 2: GPT OSS 120b -> Qwen
        # i % 4 == 3: Qwen -> Llama
        llms_old_arr = [i % len(llm_labels) for i in range(num_rewrite)]
        llms_new_arr = [(i + 1) % len(llm_labels) for i in range(num_rewrite)]

        # All rows except i%4==0 (Llama->Gemini3, already handled above)
        target_idx = [i for i in range(num_rewrite) if i % 4 != 0]

        out_path = (
            f"/share/garg/arxiv_kaggle/multillm/double_rewrite/"
            f"arxiv_{year}_ai_{category}_{subsample_size}_{rewrite_pct}_fronthalf_120b_qwen_v2.parquet"
        )

        # Retain prior work: if output file already exists, reload it and skip
        # rows whose target column already differs from the raw single-mirror value
        if os.path.exists(out_path):
            assert(False)
            prior = pd.read_parquet(out_path)
            combined = prior.copy()
            todo_idx = []
            for i in target_idx:
                target_col = llm_labels[llms_new_arr[i]]
                raw_val = arxiv_data.iloc[i][target_col]
                cur_val = combined.iloc[i][target_col]
                if pd.isna(cur_val) or cur_val == raw_val:
                    todo_idx.append(i)
            already_done = len(target_idx) - len(todo_idx)
            print(f"Year {year}: {already_done} rows already computed, {len(todo_idx)} remaining")
        else:
            todo_idx = target_idx

        if not todo_idx:
            print(f"Year {year}: all double mirrors already computed, skipping")
            continue

        ai_writing = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_idx = {
                executor.submit(
                    rewrite_abstract,
                    query_fns[llms_new_arr[i]],
                    combined.iloc[i][llm_labels[llms_old_arr[i]]],
                    llm_labels[llms_new_arr[i]],
                ): i
                for i in todo_idx
            }

            for fut in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(todo_idx),
                desc=f"Double mirrors for {year}",
            ):
                idx = future_to_idx[fut]
                ai_writing[idx] = fut.result()

        for i, (rewrite, model_name) in ai_writing.items():
            combined.at[i, model_name] = rewrite

        combined.to_parquet(out_path)
        print(f"Year {year}: saved to {out_path}")
