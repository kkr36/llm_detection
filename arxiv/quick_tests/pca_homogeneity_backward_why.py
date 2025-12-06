import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import json
import boto3
from matplotlib import pyplot as plt

def generate_embedding(input_text, bedrock, model_id="amazon.titan-embed-text-v2:0"):
    """
    Generate an embedding with the vector representation of a text input using Amazon Titan Text Embeddings G1 on demand.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        response (JSON): The embedding created by the model and the number of input tokens.
    """

    body = json.dumps({
        "inputText": input_text,
        "embeddingTypes": ["binary"]
    })

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )

    response_body = json.loads(response.get('body').read())

    return response_body['embeddingsByType']['binary']

if __name__ == "__main__":

    train_path = f"/share/garg/arxiv_kaggle/train/arxiv_tokenized_2010_ai_cs._5000.parquet"
    train_df = pd.read_parquet(train_path)

    ai_text, human_text = train_df['ai_sentence'].tolist(), [x for x in train_df["human_sentence"].tolist() if len(x) > 0]
    ai_text, human_text = ai_text[:3000], human_text[:3000]
    texts = ai_text + human_text
    texts = [" ".join(x) for x in texts]
    labels = [1 for _ in range(len(ai_text))] + [0 for _ in range(len(human_text))]

    embeddings = []
    max_workers = 10  # tune based on rate limits
    bedrock = boto3.client("bedrock-runtime", region_name='us-west-2')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_embedding, t, bedrock): i for i, t in enumerate(texts)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            i = futures[future]
            emb = future.result()
            if emb is not None:
                embeddings.append(np.array(emb))

    # Stack all embeddings
    np_df = np.vstack(embeddings)

    n_components = 2  # adjust based on data size and variance explained
    preprocess1 = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])

    preprocess2 = Pipeline([
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])

    # Fit PCA on training set only
    train_X1 = preprocess1.fit_transform(np_df)
    train_X2 = preprocess2.fit_transform(np_df)

    # plot
    plt.scatter(train_X1[:,0], train_X1[:,1], c=labels, cmap="viridis")
    plt.colorbar(label="0 is human, 1 is llm")
    plt.savefig("with_normalize.pdf", format="pdf")
    plt.clf()

    plt.scatter(train_X2[:,0], train_X2[:,1], c=labels, cmap="viridis")
    plt.colorbar(label="0 is human, 1 is llm")
    plt.savefig("without_normalize.pdf", format="pdf")
    plt.clf()

    import pdb; pdb.set_trace()
