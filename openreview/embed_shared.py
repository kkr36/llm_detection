### given a year, get all ai reviews and their paper ids, find the real reviews that correspond (if it doesn't exist, skip), merge them, sample, embed, and save ###

import json
import numpy as np
import pandas as pd
import boto3
from tqdm import tqdm
import pickle

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

    # logger.info("Generating an embedding with Amazon Titan Text Embeddings V2 model %s", model_id)

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )

    response_body = json.loads(response.get('body').read())

    return response_body

# given raw json of real reviews, turns into dict
def process_real_reviews(real_reviews):
    papers, reviews = real_reviews["papers"], real_reviews["reviews"]
    assert(len(papers) == len(reviews))

    final_dict = {}
    for i in tqdm(range(len(papers))):
        paper_info, review_info = papers[i], reviews[i]
        try:
            assert(paper_info['paper_id'] == review_info[0]['paper_id']) # assumes at least 1 review per paper
        except:
            assert(len(review_info) == 0)
            print(paper_info['paper_id'])
            continue
        paper_id = paper_info['paper_id']
        del paper_info['paper_id']
        final_dict[paper_id] = paper_info
        final_dict[paper_id]['reviews'] = [review['review'] for review in review_info]

    return final_dict


def process_reviews(real_reviews, relevant_ai_reviews):
    # convert real reviews json into dictionary where keys are paper_id, values are {title, abstract, authors, year, venue, list_of_reviews}
    processed_real_reviews = process_real_reviews(real_reviews).copy()

    # convert ai reviews into {paper_id: ai_review} dictionary
    ai_reviews_dict = {relevant_ai_reviews.iloc[i]["paper_id"] : relevant_ai_reviews.iloc[i]["ai_review"] for i in range(len(relevant_ai_reviews))}

    # keep only real reviews that have an ai review
    keys_to_keep = list(processed_real_reviews.keys())
    processed_real_reviews = {key: processed_real_reviews[key] for key in keys_to_keep if key in ai_reviews_dict}

    print(f"{len(processed_real_reviews)} papers kept")

    # add the ai review
    for ai_id in processed_real_reviews:
        processed_real_reviews[ai_id]["ai_review"] = ai_reviews_dict[ai_id]
    
    return processed_real_reviews

def join_arr(all_emb, one_emb):
    if all_emb is None:
        return one_emb
    else:
        return np.vstack([all_emb, one_emb])

def embed_reviews(all_reviews, bedrock):
    real_embs = None
    ai_embs = None
    real_reviews = []
    fake_reviews = []
    for paper_id in tqdm(all_reviews):
        review_pool = []
        for review in all_reviews[paper_id]['reviews']:
            if "No review text" not in review:
                review_pool.append(review)
        ai_review = all_reviews[paper_id]['ai_review']
        if len(review_pool) == 0: continue
        else: one_real_review = review_pool[0]
        real_reviews.append(one_real_review)
        fake_reviews.append(ai_review)
        ai_emb, real_emb = np.array(generate_embedding(ai_review, bedrock)['embeddingsByType']['binary']), np.array(generate_embedding(one_real_review, bedrock)['embeddingsByType']['binary'])
        real_embs, ai_embs = join_arr(real_embs, real_emb), join_arr(ai_embs, ai_emb)
    assert(len(real_embs) == len(ai_embs))
    assert(len(real_reviews) == len(fake_reviews))
    return np.vstack([real_embs, ai_embs]), real_reviews + fake_reviews

def main(year):
    real_file = f"/share/garg/openreview_data/iclr_{year}_simple.json"
    with open(real_file, 'r') as f:
        real_reviews = json.load(f)
    ai_reviews = pd.read_parquet("/home/kkr36/ICLR_ai_review.parquet")

    # process all reviews
    all_reviews = process_reviews(real_reviews, ai_reviews)

    # start embedding, save
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
    embedded_reviews, raw_review_text = embed_reviews(all_reviews, bedrock)
    np.save(f"/share/garg/openreview_data/all_embeddings_{year}.npy", embedded_reviews)
    with open(f"/share/garg/openreview_data/raw_reviews_{year}.pickle", 'wb') as f:
        pickle.dump(raw_review_text, f)

if __name__ == "__main__":
    years = [2018, 2019, 2020, 2021, 2022]
    for year in years:
        main(year)



