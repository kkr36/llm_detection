import boto3
import json
from botocore.exceptions import ClientError
import time
import random
import os

with open("/home/kkr36/creds.json", "r") as f:
    keys = json.load(f)

aws_key = keys["aws_key"]

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = aws_key
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

def oss_query(context, prompt, max_retries=5, base_delay=1.0):
  # Model ID
  model_id = 'openai.gpt-oss-120b-1:0'

  # Create the request body
  native_request = {
    "model": model_id, # You can omit this field
    "messages": [
      {
        "role": "system",
        "content": context + "Do not reason too long or quote the text in your reasoning."
      },
      {
        "role": "user",
        "content": prompt
      }
    ],
    "max_completion_tokens": 1000,
    "temperature": 0.5,
    "top_p": 0.9
  }

  for attempt in range(max_retries):
    try:
      # print("retrying")
      response = bedrock.invoke_model(
          modelId=model_id,
          body=json.dumps(native_request)
        )
      response_body = json.loads(response['body'].read().decode('utf-8'))
      assert(len(response_body['choices']) == 1), f"more than 1 choice somehow?? {response_body['choices']}"
      res = response_body["choices"][0]['message']['content']
      assert("</reasoning>" in res), f"no reasoning somehow?? {res}"
      # import pdb; pdb.set_trace()
      if res.split("</reasoning>")[-1] == '':
        print(res, 'AAHH BAD HAPPENED')
        assert(False), "retrying bc empty post-reasoning"
      return res.split("</reasoning>")[-1]
    except (ClientError, Exception) as e:
        wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
        print(
            f"[Attempt {attempt+1}/{max_retries}] Error invoking '{model_id}': {e}\n"
            f"Retrying in {wait_time:.2f}s..."
        )
        time.sleep(wait_time)
    except e:
        print(e)
        native_request['messages'][0]['content'] += "Reduce reasoning prior to writing."  
        wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
        print(
            f"[Attempt {attempt+1}/{max_retries}] Error invoking '{model_id}': {e}\n"
            f"Retrying in {wait_time:.2f}s..."
        )
        time.sleep(wait_time)
        
    # If all retries fail:
    print(f"ERROR: Exhausted retries for '{model_id}'. Giving up.")
    raise RuntimeError(f"Failed to invoke model '{model_id}' after {max_retries} attempts.")


from openai import OpenAI
import json
with open("/home/kkr36/creds.json", "r") as f:
    keys = json.load(f)

aws_key = keys["aws_key"]

client = OpenAI(
    base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1", 
    api_key=aws_key # Replace with actual API key
)

import re

def openai_oss_query(context, prompt, max_retries=3):
    messages = [
        {
            "role": "developer",
            "content": (
                context
                # "\nBe concise. Output your final answer directly. No preamble."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    for attempt in range(max_retries):
        completion = client.chat.completions.create(
            max_completion_tokens=1300,
            temperature=0.5,
            reasoning_effort='low',
            model="openai.gpt-oss-120b-1:0",
            messages=messages
        )

        raw = completion.choices[0].message.content

        # Strip reasoning blocks (closed and unclosed)
        res = re.sub(r'<reasoning>.*?</reasoning>', '', raw, flags=re.DOTALL)
        res = re.sub(r'<reasoning>.*$', '', res, flags=re.DOTALL)
        res = res.strip()

        if res:
            # import pdb; pdb.set_trace()
            return res

        # If empty, reasoning ate the whole budget — retry with stronger directive
        messages = [
            {
                "role": "developer",
                "content": (
                    context +
                    "\nCRITICAL: Output ONLY the final answer. Do not reason. Do not think step by step."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

    print("badbadnotgood after all retries", raw)
    return ""

# def openai_oss_query(context, prompt):
#   completion = client.chat.completions.create(
#       max_completion_tokens = 1300,
#       temperature = 0.5,
#       reasoning_effort='low',
#       model="openai.gpt-oss-20b-1:0",
#       messages=[
#           {
#               "role": "developer",
#               "content": context + "\nMINIMIZE REASONING (< 100 WORDS)."
#           },
#           {
#               "role": "user",
#               "content": prompt
#           }
#       ]
#   )
#   # import pdb; pdb.set_trace()
#   res = completion.choices[0].message.content.split("reasoning>")[-1]
#   if len(res) == 0: print("badbadnotgood", completion.choices[0].message.content)

#   return res


  # Parse and print the message for each choice in the chat completion
  # response_body = json.loads(response['body'].read().decode('utf-8'))
  # assert(len(response_body['choices']) == 1)


  # for choice in response_body['choices']:
  #     print(choice['message']['content'])
  # return response_body

# def oss_query(client, context, prompt, max_retries=5, base_delay=1.0):

#     model_id = 'openai.gpt-oss-120b-1:0'

#     native_request = {
#       "model": model_id, # You can omit this field
#       "messages": [
#         {
#           "role": "system",
#           "content": context
#         },
#         {
#           "role": "user",
#           "content": prompt
#         }
#       ],
#       "max_completion_tokens": 512,
#       "temperature": 0.5,
#       "top_p": 0.9
#     }

#     # Convert the native request to JSON.
#     request = json.dumps(native_request)

#     for attempt in range(max_retries):
#       try:
#         response = client.invoke_model(
#             modelId=model_id,
#             body=json.dumps(native_request)
#           )
#         model_response = json.loads(response["body"].read())
#         return model_response["generation"]
#       except (ClientError, Exception) as e:
#           wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
#           print(
#               f"[Attempt {attempt+1}/{max_retries}] Error invoking '{model_id}': {e}\n"
#               f"Retrying in {wait_time:.2f}s..."
#           )
#           time.sleep(wait_time)

#     # If all retries fail:
#     print(f"ERROR: Exhausted retries for '{model_id}'. Giving up.")
#     raise RuntimeError(f"Failed to invoke model '{model_id}' after {max_retries} attempts.")