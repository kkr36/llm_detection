import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pdb

if __name__ == "__main__":
    size = 500
    input_file = f"sample_embeddings_{size}.pkl"
    with open(input_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    for llm in embeddings:
        if llm == 'real': continue
        # combine, plot pca colored by cluster center
        colors = []
        all_text = []
        all_embeddings = None
        for i, key in enumerate(embeddings):
            if key not in ["real", llm]: continue
            if all_embeddings is None:
                all_embeddings = np.array([x[1] for x in embeddings[key]])
            else:
                all_embeddings = np.vstack([all_embeddings, np.array([x[1] for x in embeddings[key]])])
            colors += [0 if key == "real" else 1 for _ in range(len(embeddings[key]))]
            all_text += [x[0] for x in embeddings[key]]
        
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(all_embeddings)

        sc = plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=colors, cmap="bwr", alpha=.2)
        handles, _ = sc.legend_elements()
        plt.legend(handles, ["Real", "LLM"], title="Cluster")

        plt.title(f'PCA of human/{llm} embeddings')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

        # save as pdf
        plt.savefig(f"{size}_{llm}.pdf", bbox_inches='tight')
        plt.clf()

    
    
        # bottom_left_cluster = np.where((pca_embeddings[:,0] < -2) & (pca_embeddings[:,1] < -2))
        # right_cluster = np.where(pca_embeddings[:,0] > 2 & (pca_embeddings[:,1] > -1.6))
        # bottom_right_cluster = np.where((pca_embeddings[:,0] > 2) & (pca_embeddings[:,1] < -2))
        # top_left_cluster = np.where((pca_embeddings[:,0] < -2) & (pca_embeddings[:,1] > 2))
        # middle_left_cluster = np.where((pca_embeddings[:,0] < -2) & (pca_embeddings[:,1] > -2) & (pca_embeddings[:,1] < 1))
