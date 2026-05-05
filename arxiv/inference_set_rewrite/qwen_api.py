import os
import json
import boto3
import time
import functools

with open("/home/kkr36/creds.json", "r") as f:
    keys = json.load(f)

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = keys["aws_key"]

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def retry_with_backoff(max_retries=5, initial_delay=3, backoff_factor=2):
    """Decorator for exponential backoff on any exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print("Too many retries!")
                        raise
                    print(f"[Retry {attempt+1}/{max_retries}] Error: {e}. "
                          f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

@retry_with_backoff()
def qwen_query(context, prompt):
    model_id = 'qwen.qwen3-next-80b-a3b'

    native_request = {
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 700,
    }

    response = bedrock.invoke_model(modelId=model_id, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())
    return model_response["choices"][0]["message"]["content"]