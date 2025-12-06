import pandas as pd


years = [2018, 2019, 2020, 2021]
M = 512
K = 8

for year in years:
    best_interp_df = pd.read_csv(f"/share/garg/openreview_data/sae_{year}_{M}_{K}/best_interp.csv")
    best_interp_df = best_interp_df.rename(columns={'correlation': 'train_correlation', 'best_interpretation': 'hypothesis'}).drop('Unnamed: 0', axis=1)
    holdout_hypothesis_df = pd.read_csv(f"/share/garg/openreview_data/sae_{year}_{M}_{K}/hypotheses.csv")

    joined_df = pd.merge(best_interp_df, holdout_hypothesis_df, on='hypothesis', how='inner').drop('Unnamed: 0', axis=1).reset_index(drop=True)
    joined_df.to_csv(f"/share/garg/openreview_data/sae_{year}_{M}_{K}/joined_hypotheses.csv")