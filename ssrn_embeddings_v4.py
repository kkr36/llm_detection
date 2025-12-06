### save, for each llm, real/llm text pairings, chunked into 2 sentences each ###

import json
import numpy as np
import boto3
from tqdm import tqdm
import pickle
import pdb

def chunk_text(text, chunk_size=2):
    """
    Split text into chunks of N sentences.
    """
    # Split into sentences (simple regex for ., !, ? followed by space or end of string)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

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

    return response_body

llms = ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "gemini-2.0-flash", "gpt-4.1-2025-04-14"]
size_per_generator = float('inf')

if __name__ == "__main__":
    for llm in llms:
        print(f"starting {llm}")
        with open(f"DetectionAI/all_json/100_ai_generated_text_only_files/generated_output_{llm}.json", 'r') as f:
            llm_json = json.load(f)
    
        real_data = [x['text'] for x in llm_json]
        llm_texts = [x['ai_generated'][llm] for x in llm_json]
        sample_texts = {}

        # sample 500 from each
        effective_sample = min(size_per_generator, len(real_data))
        np.random.seed(42)
        indices_all = np.random.choice([i for i in range(len(real_data))], size=effective_sample, replace=False)
        sample_real = [real_data[i] for i in indices_all]
        sample_texts['llm'] = [llm_texts[i] for i in indices_all]
        sample_texts['real'] = sample_real

        # embed
        bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
        embeddings = {key: [] for key in sample_texts}
        
        for source in sample_texts:
            for text in tqdm(sample_texts[source]):
                embedding = generate_embedding(text, bedrock)
                embeddings[source].append((text, embedding['embeddingsByType']['binary']))
        
        # save embeddings
        try:
            with open(f"/share/garg/ssrn_data/aligned_embeddings_{effective_sample}_{llm}.pkl", 'wb') as f:
                pickle.dump(embeddings, f)
                print(f"dumped embeddings {llm} to /share/garg/ssrn_data/aligned_embeddings_{effective_sample}_{llm}.pkl")
        except:
            pdb.set_trace()


    
    