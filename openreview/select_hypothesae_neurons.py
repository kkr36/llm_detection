from hypothesaes.select_neurons import select_neurons


if __name__ == "__main__":

    years = [2018, 2019, 2020, 2021]

    for year in years:

        train_labels = [0 if i < len(train_embeddings)//2 else 1 for i in range(len(train_embeddings))]
        val_labels = [0 if i < len(val_embeddings)//2 else 1 for i in range(len(val_embeddings))]
        
        # Select neurons using different methods
        selected_neurons, scores = select_neurons(
            activations=activations,
            target=labels,
            n_select=20,
            method="correlation",  # Options: "lasso", "correlation", "separation_score"
            # Method-specific parameters:
            # For lasso:
            # classification=False,  # Whether this is a classification task, which affects the loss function (BCE vs. MSE)
            # alpha=None,  # LASSO regularization strength (None = auto-search)
            # max_iter=1000,  # Maximum iterations for solver
            
            # For separation_score:
            # n_top_activating=100,  # Number of top-activating examples to consider
            # n_zero_activating=None,  # Number of zero-activating examples (None = same as top)
        )