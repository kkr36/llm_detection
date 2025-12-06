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
print("imported")

if __name__ == "__main__":
    
    chunked = True
    want_plot = False
    select_method = "lasso" # "lasso", "correlation", "separation_score"
    chunk_size = 2
    years = [2018, 2019, 2020, 2021][:1]

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
        
        if want_plot:
            pca = PCA(n_components=2)
            iclr_pca = pca.fit_transform(embeddings)
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
            iclr_tsne = tsne.fit_transform(embeddings)

        # get_rid_frac = .5
        # get_rid_idx = int(len(embeddings)*get_rid_frac/2)
        # embeddings = embeddings[get_rid_idx:-get_rid_idx]
        # raw_text = raw_text[get_rid_idx:-get_rid_idx]
        # assert(len(raw_text) == len(embeddings))

        val_frac = .1
        train_idx = int(len(embeddings)*val_frac/2)

        M, K = 1024, 32
        prefix_lengths = [M//4, M//2, M]
        # folder in /share/garg/openreview_data where everything gets saved
        share_subfolder = f"sae_{year}_{M}_{K}{combined_conditional_str}_{select_method}"
        checkpoint_dir = f'/share/garg/openreview_data/{share_subfolder}'
        train_embeddings = embeddings[train_idx:-train_idx]
        val_embeddings = np.vstack([embeddings[:train_idx], embeddings[-train_idx:]])
        train_labels = [0 if i < len(train_embeddings)//2 else 1 for i in range(len(train_embeddings))]
        val_labels = [0 if i < len(val_embeddings)//2 else 1 for i in range(len(val_embeddings))]
        train_text, val_text = raw_text[train_idx:-train_idx], raw_text[:train_idx] + raw_text[-train_idx:]
        if want_plot:
            train_pca, val_pca = iclr_pca[train_idx:-train_idx], np.vstack([iclr_pca[:train_idx], iclr_pca[-train_idx:]])
            train_tsne, val_tsne = iclr_tsne[train_idx:-train_idx], np.vstack([iclr_tsne[:train_idx], iclr_tsne[-train_idx:]])
        print(f"Train Size: {len(train_labels)} || Val Size: {len(val_labels)}")
        


        def plot(train_arr, val_arr, vers):
            sc = plt.scatter(train_arr[:, 0], train_arr[:, 1], c=train_labels, cmap="Set2", alpha=.2)
            handles, _ = sc.legend_elements()
            plt.legend(handles, ["Real", "LLM"], title="Source")

            plt.title(f'{vers} of embeddings {year}')
            plt.xlabel(f'{vers} 1')
            plt.ylabel(f'{vers} 2')

            # save as pdf
            plt.savefig(f"logs/{vers}_{flipped_conditional_str}{year}_train.pdf", bbox_inches='tight')
            plt.clf()

            sc = plt.scatter(val_arr[:, 0], val_arr[:, 1], c=val_labels, cmap="Set2", alpha=.2)
            handles, _ = sc.legend_elements()
            plt.legend(handles, ["Real", "LLM"], title="Source")

            plt.title(f'{vers} of embeddings {year}')
            plt.xlabel(f'{vers} 1')
            plt.ylabel(f'{vers} 2')

            # save as pdf
            plt.savefig(f"logs/{vers}_{flipped_conditional_str}{year}_val.pdf", bbox_inches='tight')
            plt.clf()
            
        if want_plot:
            plot(train_tsne, val_tsne, "T-SNE")
            plot(train_pca, val_pca, "PCA")
        # import pdb; pdb.set_trace()
        # continue

        model = train_sae(
            embeddings=train_embeddings,
            M=M,
            K=K,
            matryoshka_prefix_lengths=prefix_lengths,
            batch_topk=False,  # Optional: enable Batch Top-K sparsity
            checkpoint_dir=checkpoint_dir,
            val_embeddings=val_embeddings,
            n_epochs=150,
            patience=15
        )

        # Get activations from the model
        train_activations = model.get_activations(train_embeddings)
        print(f"Neuron activations shape: {train_activations.shape}")
        # import pdb; pdb.set_trace()

        # Select neurons using different methods
        selected_neurons, scores = select_neurons(
            activations=train_activations,
            target=train_labels,
            n_select=20,
            # n_select=1,
            method=select_method,  # Options: "lasso", "correlation", "separation_score"
        )
        # import pdb; pdb.set_trace()

        # Task-specific instructions help the LLM generate better interpretations
        TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are reviews of papers for the ICLR conference.
        Features should describe a *specific* aspect of the review's writing style or content. For example:
        - "mentions temporal difference learning (TDL)"
        - "speaks in the first person"
        - "expresses doubt regarding paper results"
        """

        # Initialize the interpreter
        interpreter = NeuronInterpreter(
            interpreter_model="llama70b",  # Model for generating interpretations
            annotator_model="llama8b",  # Model for scoring interpretations
            n_workers_interpretation=1,  # Parallel workers for interpretation
            n_workers_annotation=15,  # Parallel workers for annotation
            cache_name=share_subfolder,  # Cache name for storing annotations
        )

        # Configure interpretation parameters
        interpret_config = InterpretConfig(
            sampling=SamplingConfig(
                n_examples=30,  # Number of examples to show the LLM; half are top-activating, half are zero-activating
            ),
            llm=LLMConfig(
                temperature=0.7,  # Temperature for generation
                max_interpretation_tokens=75,  # Max tokens for interpretation
            ),
            n_candidates=3,  # Generate multiple interpretations per neuron
            task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS,
        )

        # Generate interpretations for selected neurons
        interpretations = interpreter.interpret_neurons(
            texts=train_text,
            activations=train_activations,
            neuron_indices=selected_neurons,
            config=interpret_config,
        )

        import pickle
        with open(f"/share/garg/openreview_data/{share_subfolder}/interpretations.pickle", 'wb') as f:
            pickle.dump(interpretations, f)

        # import pdb; pdb.set_trace()

        # Score the interpretations to find the best ones
        scoring_config = ScoringConfig(
            n_examples=30,  # Number of examples to score each interpretation; half are top-activating, half are zero-activating
        )

        all_metrics = interpreter.score_interpretations(
            texts=train_text,
            activations=train_activations,
            interpretations=interpretations,
            config=scoring_config,
        )

        # Use the scoring results to find the best interpretation (out of the n_candidates) for each neuron
        best_interp_df = pd.DataFrame({
            'neuron_idx': selected_neurons,
            'train_correlation': scores,
            'hypothesis': [
                max(all_metrics[neuron_idx].items(), key=lambda x: x[1]['f1'])[0]
                for neuron_idx in selected_neurons
            ],
            'best_f1': [
                max(all_metrics[neuron_idx].items(), key=lambda x: x[1]['f1'])[1]['f1']
                for neuron_idx in selected_neurons
            ],
        })

        best_interp_df.to_csv(f"/share/garg/openreview_data/{share_subfolder}/best_interp.csv",)

        # import pdb; pdb.set_trace()

        # Evaluate hypotheses on a holdout set
        holdout_annotations = annotate_texts_with_concepts(
            texts=val_text,
            concepts=best_interp_df['hypothesis'].tolist(),
            cache_name=share_subfolder,
            model="llama8b",
            n_workers=15,
        )

        holdout_metrics, holdout_hypothesis_df = score_hypotheses(
            hypothesis_annotations=holdout_annotations,
            y_true=np.array(val_labels),
            classification=True,
        )

        print(f"Holdout Set Metrics:")
        print(f"R² Score: {holdout_metrics['r2']:.3f}")
        print(f"Significant hypotheses: {holdout_metrics['Significant'][0]}/{holdout_metrics['Significant'][1]} " 
            f"(p < {holdout_metrics['Significant'][2]:.3e})")
        
        holdout_hypothesis_df.to_csv(f"/share/garg/openreview_data/{share_subfolder}/hypotheses.csv")
        joined_df = pd.merge(best_interp_df, holdout_hypothesis_df, on='hypothesis', how='inner')
        joined_df.to_csv(f"/share/garg/openreview_data/{share_subfolder}/joined_hypotheses.csv")












        # # save activations
        # np.save(f"{checkpoint_dir}/activations.npy", train_activations)
        # print(f"saved activations to {checkpoint_dir}/train_activations.npy")