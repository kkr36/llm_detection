import os
import json
from tqdm import tqdm
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError
import concurrent.futures
import time
import random
import pdb

def prompt_model(bedrock, context, prompt, model_name, max_retries=5, base_delay=1.0):
    # Embed the prompt in Llama 3's instruction format.
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {context}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    # Format the request payload using the model's native structure.
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(modelId=model_name, body=request)
            model_response = json.loads(response["body"].read())
            return model_response["generation"]

        except (ClientError, Exception) as e:
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            print(
                f"[Attempt {attempt+1}/{max_retries}] Error invoking '{model_name}': {e}\n"
                f"Retrying in {wait_time:.2f}s..."
            )
            time.sleep(wait_time)

    # If all retries fail:
    print(f"ERROR: Exhausted retries for '{model_name}'. Giving up.")
    raise RuntimeError(f"Failed to invoke model '{model_name}' after {max_retries} attempts.")

    # try:
    #     # Invoke the model with the request.
    #     response = bedrock.invoke_model(modelId=model_name, body=request)

    # except (ClientError, Exception) as e:
    #     print(f"ERROR: Can't invoke '{model_name}'. Reason: {e}")
    #     exit(1)

    # # Decode the response body.
    # model_response = json.loads(response["body"].read())

    # # Extract and print the response text.
    # response_text = model_response["generation"]
    return response_text

def rewrite_abstract(bedrock, abstract, model_name="meta.llama3-70b-instruct-v1:0"):
    prompt1 = f"""
    The aim here is to reverse - engineer the author 's writing process by taking a piece of text from a paper and compressing it into a more
    concise form. This process simulates how an author might distill
    their thoughts and key points into a structured, yet not overly
    condensed form.
    Now as a first step, first summarize the goal of the text , e.g., is it
    introduction, or method, results? and then given a complete piece of
    text from a paper, reverse-engineer it into a list of bullet points.
    """
    context1 = f"Here is the text: {abstract}"
    res1 = prompt_model(bedrock, context1, prompt1, model_name)

    prompt2 = f"""
    Following the initial step of reverse-engineering the author's writing
    process by compressing a text segment from a paper, you now enter the
    second phase. Here, your objective is to expand upon the concise
    version previously crafted . This stage simulates how an author
    elaborates on the distilled thoughts and key points, enriching them
    into a detailed, structured narrative.
    Given the concise output from the previous step, your task is to develop
    it into a fully fleshed-out text.
    """
    context2 = f"Here is the writing: {res1}"
    res2 = prompt_model(bedrock, context2, prompt2, model_name)

    prompt3 = f"""
    Your task is to proofread the provided writing for grammatical accuracy.
    Ensure that the corrections introduce minimal distortion to the
    original content. Return nothing but the corrected text. If you were going to say something like "Here is the corrected text:" at the start of your output, remove it.
    """
    context3 = f"Here is the writing: {res2}"
    res3 = prompt_model(bedrock, context3, prompt3, model_name)
    return res3

if __name__ == "__main__":
    subsample_size = 5000
    category = "cs."
    arxiv_path = f"/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_{category}_{subsample_size}.json"
    with open(arxiv_path, 'rb') as f:
        arxiv_data = json.load(f)
    # train_years = ['2010', '2020']
    train_years = [str(x) for x in range(2021,2026,1)]
    # train_years = ['2024']
    

    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    for year in tqdm(train_years):
        if year in ['2010','2020','2024']:
            print(f"already saved for year {year}")
            continue
        arxiv_data[year] = arxiv_data[year][:subsample_size//2]
        # import pdb; pdb.set_trace()
        result = defaultdict(list)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_idx = {
                executor.submit(rewrite_abstract, bedrock, abstract): i
                for i, abstract in enumerate(arxiv_data[year])
            }

            iterator = tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(arxiv_data[year]),
                desc="Generating new abstracts",
            )

            ai_writing = [None] * len(arxiv_data[year])
            for fut in iterator:
                idx = future_to_idx[fut]
                ai_writing[idx] = fut.result()

            result['ai_abs'] = ai_writing
            result['human_abs'] = arxiv_data[year]
    
            with open(f"/share/garg/arxiv_kaggle/train/arxiv-metadata-oai-snapshot_{year}_ai_{category}_{subsample_size//2}.json", 'w') as f:
                json.dump(result, f)
        print(f"saved for year {year}")
    