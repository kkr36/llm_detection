import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    train_year = '2010'
    subsample_path = f"/share/garg/arxiv_kaggle/train/arxiv_tokenized_{train_year}_ai_cs._5000.parquet"
    raw_data = pd.read_parquet(subsample_path) # 2 cols, ai_sentence and human_sentence, containing rows of sentences (each sentence represented by a list of strings)

    # --- Extract non-empty rows ---
    # Each cell should be a list of tokens; we’ll filter out empty lists or NaNs.
    raw_data = raw_data.dropna(subset=["ai_sentence", "human_sentence"])
    # raw_data = raw_data[
    #     raw_data["ai_sentence"].apply(lambda x: len(x) > 0) &
    #     raw_data["human_sentence"].apply(lambda x: len(x) > 0)
    # ]

    # --- Combine ai_sentence and human_sentence into a single column ---
    # Label them (1 for AI, 0 for human)
    ai_df = pd.DataFrame({
        "text": raw_data["ai_sentence"].apply(lambda x: " ".join(x)),
        "label": 1
    })
    human_df = pd.DataFrame({
        "text": raw_data["human_sentence"].apply(lambda x: " ".join(x)),
        "label": 0
    })
    import pdb; pdb.set_trace()
    human_df = human_df[human_df["text"].str.strip() != '']

    combined_df = pd.concat([ai_df, human_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # --- Split into train/val ---
    train_df, val_df = train_test_split(combined_df, test_size=0.1, random_state=42, stratify=combined_df["label"])

    # --- Save to CSV ---
    train_df.to_csv(f"/share/garg/arxiv_kaggle/ft/train_{train_year}.csv", index=False)
    val_df.to_csv(f"/share/garg/arxiv_kaggle/ft/val_{train_year}.csv", index=False)

    print(f"Saved train_{train_year}.csv ({len(train_df)} rows) and val_{train_year}.csv ({len(val_df)} rows)")