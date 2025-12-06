years = [2020, 2023, 2025]

import pandas as pd

if __name__ == "__main__":
    for year in years:
        front_half = pd.read_parquet(f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_cs._10000_fronthalf.parquet")
        back_half = pd.read_parquet(f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_cs._10000_backhalf.parquet")

        joined = pd.concat([front_half, back_half]).reset_index().drop('index', axis=1)
        joined.to_parquet(f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_cs._20000.parquet")