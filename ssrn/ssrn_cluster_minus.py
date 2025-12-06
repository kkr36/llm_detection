import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pdb

if __name__ == "__main__":
    size = 500
    llms = ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "gemini-2.0-flash", "gpt-4.1-2025-04-14"]

    all_text = [] # list of (real, fake)
    all_embeddings = None
    colors = []

    for j, llm in enumerate(llms):

        input_file = f"aligned_embeddings_{size}_{llm}.pkl"
        with open(input_file, 'rb') as f:
            embeddings = pickle.load(f)

        # for each embedding, subtract real from ai text
        real_info, fake_info = embeddings['real'], embeddings['llm']
        assert(len(real_info) == len(fake_info))
        subtracted_embeddings = np.array([np.array(fake_info[i][1]) - np.array(real_info[i][1]) for i in range(len(real_info))])
        if all_embeddings is None:
            all_embeddings = subtracted_embeddings
        else:
            all_embeddings = np.vstack([all_embeddings, subtracted_embeddings])
        all_text += [(real_info[i][0], fake_info[i][0]) for i in range(len(real_info))]
        colors += [j for _ in range(len(real_info))]

    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(all_embeddings)

    sc = plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=colors, cmap="Set2", alpha=.2)
    handles, _ = sc.legend_elements()
    plt.legend(handles, llms, title="LLM")

    plt.title('PCA of (LLM - human) embeddings')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # save as pdf
    plt.savefig(f"minus_{size}.pdf", bbox_inches='tight')
    plt.clf()