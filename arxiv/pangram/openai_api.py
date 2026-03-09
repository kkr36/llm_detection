import functools
import time
from openai import OpenAI

client = OpenAI()

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
                    print(
                        f"[Retry {attempt+1}/{max_retries}] Error: {e}. "
                        f"Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator


@retry_with_backoff()
def call_gpt_5_4(context: str, prompt: str) -> str:

    response = client.responses.create(
        model="gpt-5.4",
        reasoning={"effort": "none"},
        input=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
    )

    return response.output_text