import pandas as pd
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

if __name__ == "__main__":
    epochs = 3

    logging_path = f"./PU_learning/logging_accuracy_TEDn/ArXiv_BERT_{epochs}"
    alphas = {0.01, 0.05, 0.1, 0.2, 0.3, 0.5}
    dfs = {}
    estimate_cols = ['Alpha', 'BBE', 'Scott', 'EN']
    # estimate_cols = ['Alpha', 'bbe', 'scott', 'en', 'avg']

    # read all numeric rows into a DataFrame
    files = sorted(os.listdir(logging_path))
    print([f.split('_')[1] for f in files])
    years = [int(f.split('_')[1]) for f in files]

    for (year, file) in tqdm(list(zip(years, files))):
        df = pd.read_csv(f"{logging_path}/{file}", skiprows=1+epochs, header=None)

        # filter
        filtered_df = df[df[0].isin(alphas)]
        filtered_df.columns = estimate_cols
        filtered_df = filtered_df.set_index("Alpha")

        print(filtered_df)
        dfs[year] = filtered_df
    
    for estimate_type in estimate_cols[1:]: # for each estimate type, get all mpes for all alphas and all years 
        all_mpes = []
        fig = plt.figure()
        ax = plt.subplot()

        for alpha in alphas:
            # import pdb; pdb.set_trace()
            y_axis = [dfs[year].loc[alpha, estimate_type] for year in years]
            plt.plot(years, y_axis, label=alpha)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=6, title='Alpha')
        plt.tight_layout()
        plt.xlabel("Year")
        plt.ylabel("% Pred LLM")
        # plt.savefig(f"tedn_figs_big/test_alpha_{estimate_type}_pn.pdf", bbox_inches="tight", format='pdf')
        plt.savefig(f"tedn_figs_big/test_alpha_{estimate_type}_tedn.pdf", bbox_inches="tight", format='pdf')

        plt.clf()


    