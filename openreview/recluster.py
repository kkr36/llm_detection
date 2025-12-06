from HypotheSAEs.hypothesaes.quickstart import train_sae
from HypotheSAEs.hypothesaes.select_neurons import select_neurons
from HypotheSAEs.hypothesaes.interpret_neurons import NeuronInterpreter, InterpretConfig, SamplingConfig, LLMConfig, ScoringConfig
from HypotheSAEs.hypothesaes.annotate import annotate_texts_with_concepts
from HypotheSAEs.hypothesaes.evaluation import score_hypotheses

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if __name__ == "__main__":
    
    chunked = False
    chunk_size = 2
    years = [2018, 2019, 2020, 2021][2:]

    for year in years:
        # embeddings = np.load(f"/share/garg/openreview_data/all_embeddings_{year}.npy")
        # with open(f"/share/garg/openreview_data/raw_reviews_{year}.pickle", 'rb') as f:
        #     raw_text = pickle.load(f)

        conditional_str = f"_{chunk_size}" if chunked else ''
        flipped_conditional_str = f"chunked_" if chunked else ''
        combined_conditional_str = f"_chunked_{chunk_size}" if chunked else ''
        embeddings = np.load(f"/share/garg/openreview_data/{flipped_conditional_str}all_embeddings_{year}{conditional_str}.npy")
        with open(f"/share/garg/openreview_data/{flipped_conditional_str}raw_reviews_{year}{conditional_str}.pickle", 'rb') as f:
            raw_text = pickle.load(f)

        # get_rid_frac = .5
        # get_rid_idx = int(len(embeddings)*get_rid_frac/2)
        # embeddings = embeddings[get_rid_idx:-get_rid_idx]
        # raw_text = raw_text[get_rid_idx:-get_rid_idx]
        # assert(len(raw_text) == len(embeddings))

        val_frac = .1
        train_idx = int(len(embeddings)*val_frac/2)

        M, K = 1024, 64
        prefix_lengths = [M//4, M//2, M]
        checkpoint_dir = f'/share/garg/openreview_data/sae_{year}_{M}_{K}{combined_conditional_str}'
        train_embeddings = embeddings[train_idx:-train_idx]
        val_embeddings = np.vstack([embeddings[:train_idx], embeddings[-train_idx:]])
        train_labels = [0 if i < len(train_embeddings)//2 else 1 for i in range(len(train_embeddings))]
        val_labels = [0 if i < len(val_embeddings)//2 else 1 for i in range(len(val_embeddings))]
        train_text, val_text = raw_text[train_idx:-train_idx], raw_text[:train_idx] + raw_text[-train_idx:]
        # train_pca, val_pca = iclr_pca[train_idx:-train_idx], np.vstack([iclr_pca[:train_idx], iclr_pca[-train_idx:]])
        # train_tsne, val_tsne = iclr_tsne[train_idx:-train_idx], np.vstack([iclr_tsne[:train_idx], iclr_tsne[-train_idx:]])
        # print(f"Train Size: {len(train_pca)} || Val Size: {len(val_pca)}")

        model = train_sae(
            embeddings=train_embeddings,
            M=M,
            K=K,
            matryoshka_prefix_lengths=prefix_lengths,
            batch_topk=False,  # Optional: enable Batch Top-K sparsity
            checkpoint_dir=checkpoint_dir,
            val_embeddings=val_embeddings,
            n_epochs=110,
            patience=15
        )

        # Get activations from the model
        train_activations = model.get_activations(train_embeddings)
        val_activations = model.get_activations(val_embeddings)
        print(f"Neuron activations shape: {train_activations.shape}")

        # TODO isolate the neurons found in previous search
        hypothesis_df = pd.read_csv(f"{checkpoint_dir}/joined_hypotheses.csv")
        selected_neurons = hypothesis_df["neuron_idx"].tolist()
        # import pdb; pdb.set_trace()
        train_relevant, val_relevant = train_activations[:,selected_neurons], val_activations[:,selected_neurons]

        for neurons, labels, label in [(train_relevant, train_labels, "train"), (val_relevant, val_labels, "val")]:
            pca = PCA(n_components=2)
            iclr_pca = pca.fit_transform(neurons)
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
            iclr_tsne = tsne.fit_transform(neurons)

            def plot(arr, labels, vers):
                sc = plt.scatter(arr[:, 0], arr[:, 1], c=labels, cmap="Set2", alpha=.2)
                handles, _ = sc.legend_elements()
                plt.legend(handles, ["Real", "LLM"], title="Source")

                plt.title(f'{vers} of embeddings {year}')
                plt.xlabel(f'{vers} 1')
                plt.ylabel(f'{vers} 2')

                # save as pdf
                plt.savefig(f"logs/recluster/{vers}_{flipped_conditional_str}{year}_{label}.pdf", bbox_inches='tight')
                plt.clf()

            plot(iclr_tsne, labels, "T-SNE")
            plot(iclr_pca, labels, "PCA")