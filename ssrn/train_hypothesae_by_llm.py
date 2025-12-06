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

if __name__ == "__main__":

    llms = ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "gemini-2.0-flash", "gpt-4.1-2025-04-14"][-1:]
    size_per_generator = 1000

    for llm in llms:
        with open(f"/share/garg/ssrn_data/aligned_embeddings_{size_per_generator}_{llm}.pkl", 'rb') as f:
            data = pickle.load(f) 
        # process into embeddings and raw text
        embeddings_llm, embeddings_real = np.array([data['llm'][i][1] for i in range(len(data['llm']))]), np.array([data['real'][i][1] for i in range(len(data['real']))])
        text_llm, text_real = [data['llm'][i][0] for i in range(len(data['llm']))], [data['real'][i][0] for i in range(len(data['real']))]
        assert(len(data['real']) == len(data['llm']))

        embeddings = np.vstack([embeddings_real, embeddings_llm])
        raw_text = text_real + text_llm
        assert(len(raw_text) == len(embeddings))

        pca = PCA(n_components=2)
        iclr_pca = pca.fit_transform(embeddings)

        # get_rid_frac = .5
        # get_rid_idx = int(len(embeddings)*get_rid_frac/2)
        # embeddings = embeddings[get_rid_idx:-get_rid_idx]
        # raw_text = raw_text[get_rid_idx:-get_rid_idx]
        # assert(len(raw_text) == len(embeddings))

        val_frac = .2
        train_idx = int(len(embeddings)*val_frac/2)

        M, K = 512, 16
        prefix_lengths = [M//8, M]
        checkpoint_dir = f'/share/garg/ssrn_data/sae_{llm}_{M}_{K}_class'
        train_embeddings = embeddings[train_idx:-train_idx]
        val_embeddings = np.vstack([embeddings[:train_idx], embeddings[-train_idx:]])
        train_labels = [0 if i < len(train_embeddings)//2 else 1 for i in range(len(train_embeddings))]
        val_labels = [0 if i < len(val_embeddings)//2 else 1 for i in range(len(val_embeddings))]
        train_text, val_text = raw_text[train_idx:-train_idx], raw_text[:train_idx] + raw_text[-train_idx:]
        train_pca, val_pca = iclr_pca[train_idx:-train_idx], np.vstack([iclr_pca[:train_idx], iclr_pca[-train_idx:]])
        assert(len(train_labels) == len(train_embeddings) and len(val_labels) == len(val_embeddings))

        def plot():
            sc = plt.scatter(train_pca[:, 0], train_pca[:, 1], c=train_labels, cmap="Set2", alpha=.2)
            handles, _ = sc.legend_elements()
            plt.legend(handles, ["Real", "LLM"], title="Source")

            plt.title(f'PCA of embeddings {llm}')
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')

            # save as pdf
            plt.savefig(f"logs/{llm}_train.pdf", bbox_inches='tight')
            plt.clf()

            sc = plt.scatter(val_pca[:, 0], val_pca[:, 1], c=val_labels, cmap="Set2", alpha=.2)
            handles, _ = sc.legend_elements()
            plt.legend(handles, ["Real", "LLM"], title="Source")

            plt.title(f'PCA of embeddings {llm}')
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')

            # save as pdf
            plt.savefig(f"logs/{llm}_val.pdf", bbox_inches='tight')
            plt.clf()
        plot()
        # continue

        model = train_sae(
            embeddings=train_embeddings,
            M=M,
            K=K,
            matryoshka_prefix_lengths=prefix_lengths,
            batch_topk=False,  # Optional: enable Batch Top-K sparsity
            checkpoint_dir=checkpoint_dir,
            val_embeddings=val_embeddings,
            n_epochs=110,
        )

        # Get activations from the model
        train_activations = model.get_activations(train_embeddings)
        print(f"Neuron activations shape: {train_activations.shape}")

        # Select neurons using different methods
        selected_neurons, scores = select_neurons(
            activations=train_activations,
            target=train_labels,
            n_select=20,
            # n_select=1,
            method="correlation",  # Options: "lasso", "correlation", "separation_score"
        )
        # import pdb; pdb.set_trace()

        # Task-specific instructions help the LLM generate better interpretations
        TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are texts from a Huggingface dataset.
        Features should describe a *specific* aspect of the text's writing style or content. For example:
        - "set in Victorian England"
        - "speaks in the first person"
        - "speaks in a formal tone"
        """

        # Initialize the interpreter
        interpreter = NeuronInterpreter(
            interpreter_model="llama70b",  # Model for generating interpretations
            annotator_model="llama8b",  # Model for scoring interpretations
            n_workers_interpretation=1,  # Parallel workers for interpretation
            n_workers_annotation=15,  # Parallel workers for annotation
            cache_name=f"sae_{llm}_{M}_{K}_class",  # Cache name for storing annotations
        )

        # Configure interpretation parameters
        interpret_config = InterpretConfig(
            sampling=SamplingConfig(
                n_examples=20,  # Number of examples to show the LLM; half are top-activating, half are zero-activating
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
        # import pdb; pdb.set_trace()

        # Evaluate hypotheses on a holdout set
        holdout_annotations = annotate_texts_with_concepts(
            texts=val_text,
            concepts=best_interp_df['hypothesis'].tolist(),
            cache_name=f"sae_{llm}_{M}_{K}_class",
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

        ### DUMP EVERYTHING ###
        import pickle
        with open(f"/share/garg/ssrn_data/sae_{llm}_{M}_{K}_class/interpretations.pickle", 'wb') as f:
            pickle.dump(interpretations, f)

        best_interp_df.to_csv(f"/share/garg/ssrn_data/sae_{llm}_{M}_{K}_class/best_interp.csv",)

        holdout_hypothesis_df.to_csv(f"/share/garg/ssrn_data/sae_{llm}_{M}_{K}_class/hypotheses.csv")
        joined_df = pd.merge(best_interp_df, holdout_hypothesis_df, on='hypothesis', how='inner')
        joined_df.to_csv(f"/share/garg/ssrn_data/sae_{llm}_{M}_{K}_class/joined_hypotheses.csv")
