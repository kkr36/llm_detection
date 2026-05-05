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

with open("/home/kkr36/creds.json", "r") as f:
    keys = json.load(f)

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = keys["aws_key"]

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

def llama_query(context, prompt, max_retries=5, base_delay=1.0):
    # model_id = 'meta.llama3-3-70b-instruct-v1:0'
    model_id = 'arn:aws:bedrock:us-west-2:767397830210:inference-profile/us.meta.llama3-3-70b-instruct-v1:0'

    # Embed the prompt in Llama 3's instruction format.
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {context}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    # Format the request payload using the model's native structure.
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 700,
        "temperature": 0.5,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(modelId=model_id, body=request)
            model_response = json.loads(response["body"].read())
            return model_response["generation"]

        except (ClientError, Exception) as e:
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            print(
                f"[Attempt {attempt+1}/{max_retries}] Error invoking '{model_id}': {e}\n"
                f"Retrying in {wait_time:.2f}s..."
            )
            time.sleep(wait_time)

    # If all retries fail:
    print(f"ERROR: Exhausted retries for '{model_id}'. Giving up.")
    raise RuntimeError(f"Failed to invoke model '{model_id}' after {max_retries} attempts.")