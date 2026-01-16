from google import genai
from google.genai import types
import time
import functools
import json

with open("/home/kkr36/creds.json", "r") as f:
    keys = json.load(f)

gemini_key = keys["gemini_api_key"]

client = genai.Client(api_key = gemini_key)

def retry_with_backoff(max_retries=5, initial_delay=1, backoff_factor=2):
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
                        raise  # re-raise last exception
                    print(f"[Retry {attempt+1}/{max_retries}] Error: {e}. "
                          f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

@retry_with_backoff()
def call_gemini_2(context: str, prompt: str) -> str:

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            max_output_tokens=700,
            temperature=0.5,
            system_instruction=context,
            thinking_config=types.ThinkingConfig(thinking_budget=0)),
        contents=prompt
    )

    return response.text

@retry_with_backoff()
def call_gemini_2_pro(context: str, prompt: str, 
                 model: str = "gemini-2.5-pro", 
                ) -> str:

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            max_output_tokens=700,
            temperature=0.5,
            system_instruction=context,
            thinking_config=types.ThinkingConfig(thinking_budget=128)
        ),
        contents=prompt
    )
    # import pdb; pdb.set_trace()
    return response.text

@retry_with_backoff()
def call_gemini_3(context: str, prompt: str, 
                 model: str = "gemini-3-pro-preview", 
                 thinking_level: types.ThinkingLevel = types.ThinkingLevel.LOW
                ) -> str:

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            max_output_tokens=700,
            temperature=0.5,
            system_instruction=context,
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking_level
            )
        ),
        contents=prompt
    )
    return response.text

@retry_with_backoff()
def call_gemini_2_flash_lite(context: str, prompt: str) -> str:
    """
    Gemini 2.0 Flash-Lite: ultra-low latency, no thinking.
    Best for cheap / fast generations.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        config=types.GenerateContentConfig(
            max_output_tokens=700,
            temperature=0.5,
            system_instruction=context,
            # Flash-Lite does not support thinking; omit or force 0
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        contents=prompt,
    )

    return response.text


@retry_with_backoff()
def call_gemini_2_flash(
    context: str,
    prompt: str,
    max_output_tokens: int = 700,
    temperature: float = 0.5,
) -> str:
    """
    Gemini 2.0 Flash: fast + higher quality than Flash-Lite.
    No thinking / reasoning budget.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            system_instruction=context,
            # Flash models do not support thinking
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        contents=prompt,
    )

    return response.text
