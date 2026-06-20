from pangram import Pangram
import json
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

with open("/home/kkr36/creds.json", 'r') as handle:
    pangram_api_key = json.load(handle)['pangram_api_key']
pangram_client = Pangram(api_key=pangram_api_key)


def predict_with_backoff(pangram_client, text, max_retries=5, initial_delay=1):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            result = pangram_client.predict(text)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    return {}


existing_df = pd.read_csv("results_everything_100_2010.csv")
gpt_rows = existing_df[existing_df['source'] == 'ChatGPT 5.4'].copy()
texts = gpt_rows['text'].tolist()

new_rows = {
    "year": [],
    "idx": [],
    "source": [],
    "fraction_ai": [],
    "fraction_ai_assisted": [],
    "fraction_human": [],
    "num_ai_segments": [],
    "window_labels": [],
    "window_ai_assistance_scores": [],
    "window_confidences": [],
    "text": []
}

failed_requests = []

for i, (orig_idx, text) in enumerate(tqdm(zip(gpt_rows['idx'].tolist(), texts), total=len(texts), desc="ChatGPT 5.4 rerun")):
    result = predict_with_backoff(pangram_client, text)

    if result is None or len(result) == 0:
        print(f"Skipping idx={orig_idx}")
        failed_requests.append(orig_idx)
        continue

    try:
        fraction_ai = result.get('fraction_ai', np.nan)
        fraction_ai_assisted = result.get('fraction_ai_assisted', np.nan)
        fraction_human = result.get('fraction_human', np.nan)
        num_ai_segments = result.get('num_ai_segments', np.nan)

        window_labels = []
        window_ai_assistance_scores = []
        window_confidences = []

        for window in result.get('windows', []):
            window_labels.append(window.get('label', None))
            window_ai_assistance_scores.append(window.get('ai_assistance_score', np.nan))
            window_confidences.append(window.get('confidence', np.nan))

        new_rows['year'].append(gpt_rows.iloc[i]['year'])
        new_rows['idx'].append(orig_idx)
        new_rows['source'].append('ChatGPT 5.4 new')
        new_rows['fraction_ai'].append(fraction_ai)
        new_rows['fraction_ai_assisted'].append(fraction_ai_assisted)
        new_rows['fraction_human'].append(fraction_human)
        new_rows['num_ai_segments'].append(num_ai_segments)
        new_rows['window_labels'].append(window_labels)
        new_rows['window_ai_assistance_scores'].append(window_ai_assistance_scores)
        new_rows['window_confidences'].append(window_confidences)
        new_rows['text'].append(text)

    except Exception as e:
        print(f"Failed to process result for idx={orig_idx}: {e}")
        failed_requests.append(orig_idx)
        continue

new_df = pd.DataFrame(new_rows)
combined_df = pd.concat([existing_df, new_df], ignore_index=True)
# combined_df.to_csv("results_everything_100_2010_4242026.csv", index=False)
combined_df.to_csv("results_everything_100_2010_5142026.csv", index=False)


print(f"\nSaved {len(new_df)} new rows to results_everything_100_2010_5142026.csv")
if failed_requests:
    print(f"Failed requests ({len(failed_requests)}): {failed_requests}")
