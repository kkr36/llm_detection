### given embeddings for n real/ai reviews, embed them and save the image ###

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pickle
import pdb

if __name__ == "__main__":
    years = [2018, 2019, 2020, 2021]

    for year in years:
        embeddings_file = f"/share/garg/openreview_data/all_embeddings_{year}.npy"
        raw_file = f"/share/garg/openreview_data/raw_reviews_{year}.pickle"
        embeddings = np.load(embeddings_file)
        with open(raw_file, 'rb') as f:
            raw_reviews = pickle.load(f)
        real_embeddings, fake_embeddings = embeddings[:len(embeddings)//2], embeddings[len(embeddings)//2:]

        # 2d pca and plot, color
        pca = PCA(n_components=2)
        iclr_pca = pca.fit_transform(embeddings)
        left = np.where(iclr_pca[:,0] < -4)
        right = np.where(iclr_pca[:,0] > 4)
        top = np.where(iclr_pca[:,1] > 4)
        bottom = np.where(iclr_pca[:,1] < -4)
        pdb.set_trace()
        colors = [0 for _ in range(len(embeddings)//2)] + [1 for _ in range(len(embeddings)//2)]

        sc = plt.scatter(iclr_pca[:, 0], iclr_pca[:, 1], c=colors, cmap="Set2", alpha=.2)
        handles, _ = sc.legend_elements()
        plt.legend(handles, ["Real", "LLM"], title="Source")

        plt.title(f'PCA of embeddings {year}')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

        # save as pdf
        plt.savefig(f"logs/{year}.pdf", bbox_inches='tight')
        plt.clf()