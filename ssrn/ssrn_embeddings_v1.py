### sample random texts for each llm ###

import json
import numpy as np
import boto3
from tqdm import tqdm
import pickle
import pdb

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

llms = ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "gemini-2.0-flash", "gpt-4.1-2025-04-14"]
size_per_generator = 500

if __name__ == "__main__":
    with open("DetectionAI/original_corpus/original_corpus.json", 'r') as f:
        real_json = json.load(f)
    llm_jsons = {}
    for llm in llms:
        with open(f"DetectionAI/all_json/100_ai_generated_text_only_files/generated_output_{llm}.json", 'r') as f:
            llm_jsons[llm] = json.load(f)
    
    real_data = [x['text'] for x in real_json]
    llm_texts = {
        llm: [x['ai_generated'][llm] for x in llm_jsons[llm]] for llm in llms
    }
    min_size = min([len(real_data)] + [len(llm_texts[llm]) for llm in llm_texts])
    sample_texts = {}

    # sample 500 from each
    np.random.seed(42)
    indices_all = np.random.choice([i for i in range(min_size)], size=size_per_generator, replace=False)
    sample_real = [real_data[i] for i in indices_all]
    for llm in llms:
        indices_all = np.random.choice([i for i in range(min_size)], size=size_per_generator, replace=False)
        sample_texts[llm] = [llm_texts[llm][i] for i in indices_all]
    sample_texts['real'] = sample_real
    pdb.set_trace()

    # embed
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
    embeddings = {key: [] for key in sample_texts}
    
    for source in sample_texts:
        for text in tqdm(sample_texts[source]):
            embedding = generate_embedding(text, bedrock)
            embeddings[source].append((text, embedding['embeddingsByType']['binary']))
    
    # save embeddings
    try:
        with open(f"sample_embeddings_{size_per_generator}_identical.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
    except:
        pdb.set_trace()


    
    