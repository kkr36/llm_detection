from pangram import Pangram
import json
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

with open("/home/kkr36/creds.json", 'r') as handle:
    pangram_api_key = json.load(handle)['pangram_api_key']
pangram_client = Pangram(api_key=pangram_api_key)

years = [2010]
start_idx, end_idx = 0, 100
cols = ['ChatGPT 5.4']
final_dict = {
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

def predict_with_backoff(pangram_client, text, max_retries=5, initial_delay=1):
    """
    Call pangram_client.predict with exponential backoff retry logic.
    
    Args:
        pangram_client: The Pangram client instance
        text: Text to predict on
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles with each retry)
    
    Returns:
        API response dict or None if all retries failed
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            result = pangram_client.predict(text)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            
            # Exponential backoff
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Double the delay for next attempt
    
    return {}


# Add after pangram_client initialization
failed_requests = []

for year in years:
    data_path = f"gpt_54_100_2010.parquet"
    data = pd.read_parquet(data_path)
        
    for col in cols:
        texts = data[col].dropna().tolist()
        texts = texts[start_idx:end_idx]
        
        for idx, text in enumerate(tqdm(texts, desc=f"{year}-{col}")):
            result = predict_with_backoff(pangram_client, text)
            
            if result is None or len(result) == 0:
                # API call failed after all retries
                print(f"Skipping: year={year}, col={col}, idx={idx}")
                failed_requests.append((year, col, idx))
                continue
            
            try:
                # Safe dictionary access with NaN for numeric fields
                fraction_ai = result.get('fraction_ai', np.nan)
                fraction_ai_assisted = result.get('fraction_ai_assisted', np.nan)
                fraction_human = result.get('fraction_human', np.nan)
                num_ai_segments = result.get('num_ai_segments', np.nan)
                
                window_labels = []
                window_ai_assistance_scores = []
                window_confidences = []
                
                for window in result.get('windows', []):
                    window_labels.append(window.get('label', None))  # None for missing strings
                    window_ai_assistance_scores.append(window.get('ai_assistance_score', np.nan))
                    window_confidences.append(window.get('confidence', np.nan))
                
                final_dict['year'].append(year)
                final_dict['text'].append(text)
                final_dict['idx'].append(idx)
                final_dict['source'].append(col)
                final_dict['fraction_ai'].append(fraction_ai)
                final_dict['fraction_ai_assisted'].append(fraction_ai_assisted)
                final_dict['fraction_human'].append(fraction_human)
                final_dict['num_ai_segments'].append(num_ai_segments)
                final_dict['window_labels'].append(window_labels)
                final_dict['window_ai_assistance_scores'].append(window_ai_assistance_scores)
                final_dict['window_confidences'].append(window_confidences)

                if idx == 0: import pdb; pdb.set_trace()
                                
            except Exception as e:
                print(f"Failed to process result for year={year}, col={col}, idx={idx}: {e}")
                failed_requests.append((year, col, idx))
                continue

# Save results and failed requests
existing_data = pd.read_parquet("results_0_50.parquet")
df_results = pd.DataFrame(final_dict)

import pdb; pdb.set_trace()

df_results = pd.concat([existing_data, df_results])

df_results.to_parquet(f'results_54_{start_idx}_{end_idx}.parquet')
if failed_requests:
    print(f"\nFailed requests: {len(failed_requests)}")
    print(failed_requests)
                    
