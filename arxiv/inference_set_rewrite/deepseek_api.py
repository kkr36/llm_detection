import json
import time
import random
import boto3
from botocore.exceptions import ClientError

# Create Bedrock client OUTSIDE the function
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

"deepseek.v3-v1:0"

def deepseek_query(context, prompt,
                   model_id="us.deepseek.r1-v1:0",
                   max_retries=5,
                   base_delay=1.0,
                   temperature=0.5,
                   max_tokens=4096):
    """
    Query DeepSeek-R1 using Bedrock Converse API.
    Returns the assistant's text output (reasoning removed).
    """

    # Build system prompts and user messages
    system_prompts = [{"text": context}]
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]

    inference_config = {
        "temperature": temperature,
        "maxTokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            response = bedrock.converse(
                modelId=model_id,
                system=system_prompts,
                messages=messages,
                inferenceConfig=inference_config,
            )

            # Extract assistant message
            output_msg = response["output"]["message"]

            cleaned_contents = []
            for c in output_msg["content"]:
                if "text" in c:
                    cleaned_contents.append(c)
                # Skip reasoningContent
            output_msg["content"] = cleaned_contents

            # Return final assistant text
            final_text = "".join(c["text"] for c in cleaned_contents)
            return final_text.strip()

        except (ClientError, Exception) as e:
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            print(
                f"[Attempt {attempt+1}/{max_retries}] Error invoking DeepSeek-R1: {e}\n"
                f"Retrying in {wait_time:.2f}s..."
            )
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to call DeepSeek-R1 after {max_retries} attempts.")
