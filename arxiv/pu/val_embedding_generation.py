import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import boto3
from botocore.exceptions import ClientError


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
    years = list(range(2015, 2026, 2))

    for train_year in tqdm(years):
        filename = f"arxiv_tokenized_{train_year}_val_cs._5000"
        train_path = f"/share/garg/arxiv_kaggle/val/{filename}.parquet"
        train_df = pd.read_parquet(train_path)
        train_df = pd.DataFrame({
                "text": train_df["inference_sentence"].apply(lambda x: " ".join(x)),
            })
        
        texts = train_df["text"].tolist()
        labels = [-1 for _ in range(len(texts))]

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
        np.save(f"/share/garg/arxiv_kaggle/pu/val_{train_year}.npy", np_df)
