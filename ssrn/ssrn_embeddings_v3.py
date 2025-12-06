### save, for each real text, all texts from the llms for the same generation ###

import json
import numpy as np
import boto3
from tqdm import tqdm
import pickle
import pdb

def generate_embedding(input_text, bedrock, model_id="amazon.titan-embed-text-v2:0"):
    """
    Generate an embedding with the vector representation of a text input using Amazon Titan Text Embeddings.
    """
    body = json.dumps({
        "inputText": input_text,
        "embeddingTypes": ["binary"]
    })

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    return response_body['embeddingsByType']['binary']

llms = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "gemini-2.0-flash",
    "gpt-4.1-2025-04-14"
]
size_per_generator = float('inf')

if __name__ == "__main__":
    # Load all jsons
    llm_outputs = {}
    real_data = None
    for llm in llms:
        with open(f"DetectionAI/all_json/100_ai_generated_text_only_files/generated_output_{llm}.json", 'r') as f:
            llm_json = json.load(f)

        if real_data is None:
            real_data = [x['text'] for x in llm_json]

        llm_outputs[llm] = [x['ai_generated'][llm] for x in llm_json]

    # Sample indices
    effective_sample = min(size_per_generator, len(real_data))
    np.random.seed(42)
    indices_all = np.random.choice(len(real_data), size=effective_sample, replace=False)

    sample_real = [real_data[i] for i in indices_all]
    sample_llms = {llm: [llm_outputs[llm][i] for i in indices_all] for llm in llms}

    # Embed
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

    grouped_data = {}
    for idx in tqdm(range(effective_sample)):
        real_text = sample_real[idx]
        llm_texts = {llm: sample_llms[llm][idx] for llm in llms}

        try:
            # Embed real text
            real_emb = generate_embedding(real_text, bedrock)

            # Embed LLM texts
            llm_embs = {llm: generate_embedding(text, bedrock) for llm, text in llm_texts.items()}

            # Store in dictionary
            grouped_data[real_text] = {
                "real": (real_text, real_emb),
                **{llm: (text, llm_embs[llm]) for llm, text in llm_texts.items()}
            }

        except Exception as e:
            print(f"Error embedding index {idx}: {e}")
            pdb.set_trace()

    # Save
    save_path = f"/share/garg/ssrn_data/aligned_embeddings_{effective_sample}_dict.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(grouped_data, f)
        print(f"dumped embeddings to {save_path}")

    