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
    train_year = 2010
    filename = f"val_{train_year}"
    train_path = f"/share/garg/arxiv_kaggle/ft/{filename}.csv"
    train_df = pd.read_csv(train_path)

    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()
    import pdb; pdb.set_trace()

    embeddings = []
    max_workers = 10  # tune based on rate limits
    bedrock = boto3.client("bedrock-runtime", region_name='us-west-2')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_embedding, t, bedrock): i for i, t in enumerate(texts)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            i = futures[future]
            emb = future.result()
            if emb is not None:
                embeddings.append(np.array(emb + [labels[i]]))

    # Stack all embeddings
    np_df = np.vstack(embeddings)

    # # for each row, get embeddings, append label, and add to np array of embeddings
    # np_df = None
    # bedrock = boto3.client("bedrock-runtime", region_name='us-west-2')
    # for i in tqdm(list(range(len(train_df)))):
    #     row = train_df.iloc[i]
    #     text, label = row['text'], row['label']

    #     embedding = np.array(generate_embedding(text, bedrock)['embeddingsByType']['binary'] + [label])
    #     if np_df is None:
    #         np_df = embedding
    #     else:
    #         np_df = np.vstack([np_df, embedding])
    
    # save to pu set
    # import pdb; pdb.set_trace()
    np.save(f"/share/garg/arxiv_kaggle/pu/{filename}_train.npy", np_df)
