### DO NOT OVERWRITE, EDIT, OR TOUCH ANYTHING IN THIS FILE ###
# must be run using conda env *llm_master*!

import os
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from openai_api import openai_oss_query
from gemini_api import call_gemini_2_pro
from util import individual_predict, split_into_sentences, clean_text, predict_with_backoff
from strategy import CURRENT_STRATEGY, CURRENT_TIMESTEP
from pathlib import Path
import numpy as np
import sys
import torch
sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_helper import get_model
from pangram import Pangram
import time

with open("/home/kkr36/creds.json", 'r') as handle:
    pangram_api_key = json.load(handle)['pangram_api_key']
pangram_client = Pangram(api_key=pangram_api_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

query_fns = [openai_oss_query]
llm_labels = ["GPT OSS 120b"]
assert(len(query_fns) == len(llm_labels))

split = "val"
to_rewrite = 15
pangram = False
output_csv = f"results_{CURRENT_TIMESTEP}_oss_{split}_{to_rewrite}.csv"

if __name__ == "__main__":
    print(f"starting generation for t={CURRENT_TIMESTEP}")
    ### FIXED PARAMS
    subsample_size = 20000
    category = "cs."
    train_year = 2010

    old_text = [] # holds llm mirrors from a previous iteration, t=-1
    new_text = [] # holds llm mirrors from current iteration, CURRENT_STRATEGY / CURRENT_TIMESTEP
    human_text = [] # holds human-written abstracts

    val_start = 2500
    test_start = 2600

    start_idx = val_start if split == "val" else test_start
    # assert val_start + to_rewrite <= test_start if split == "val" else True
    assert to_rewrite == 15 if split == "val" else 50

    fpath = f"results_0_oss_val_15.csv" if split == "val" else "results_0_oss_test_50.csv"
    arxiv_data = pd.read_csv(fpath)
    orig_abstracts = arxiv_data["mirror_0"].tolist()
    rng = np.random.default_rng(seed=CURRENT_TIMESTEP)
    # rng.shuffle(orig_abstracts)
    llms = [i % len(llm_labels) for i in range(len(arxiv_data))]

    # generate to_rewrite mirrors
    print("generating mirrors")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}

    def process_row(i):
        # row = arxiv_data.iloc[i]
        # orig_abs = row['human_abstract']
        orig_abs = orig_abstracts[i]
        new_abs, _ = CURRENT_STRATEGY(query_fns[llms[i]], orig_abs, llm_labels[llms[i]])
        return i, new_abs, orig_abs

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, i): i for i in range(to_rewrite)}
        for future in tqdm(as_completed(futures), total=to_rewrite):
            i, new_abs, orig_abs = future.result()
            results[i] = (new_abs, orig_abs)

    # Preserve original order
    for i in sorted(results):
        new_text.append(results[i][0])
        human_text.append(results[i][1])

    # for i in tqdm(range(val_start, val_start + to_rewrite)):
    #     row = arxiv_data.iloc[i]
    #     orig_abs = row['mirror_0']
    #     # llm_abs = row[llm_labels[llms[i]]]
    #     # assert llm_abs is not None and len(llm_abs) > 0 # make sure the old mirror exists

    #     new_abs, _ = CURRENT_STRATEGY(query_fns[llms[i]], orig_abs, llm_labels[llms[i]])

    #     # old_text.append(llm_abs)
    #     new_text.append(new_abs)
    #     human_text.append(orig_abs)
    
    # pass through pre-trained classification model -- higher numbers = lower P(human); we want to MINIMIZE the scores of this classifier
    # specifically, the average score over new_text should be lower than the average score over old_text (meaning CURRENT_STRATEGY is better than the original), and approach average score over human abstracts
    # each row of abstract_dict represents sentences of one human abstract, the sentences of its llm mirror, and scores per sentence
    abstract_dict = {
        "original": [],
        "mirroring_llm": [],
        f"mirror_{CURRENT_TIMESTEP}": [],
    }
    
    for i, (human_abstract, mirror_t) in enumerate(list(zip(human_text, new_text))):
        abstract_dict["original"].append(human_abstract)
        abstract_dict[f"mirror_{CURRENT_TIMESTEP}"].append(mirror_t)
        abstract_dict["mirroring_llm"].append(llm_labels[llms[i]])

    # import pdb; pdb.set_trace()

    if pangram:
        failed_requests = []
        final_dict = {
            "fraction_ai": [],
            "fraction_ai_assisted": [],
            "fraction_human": [],
            "num_ai_segments": [],
            "window_labels": [],
            "window_ai_assistance_scores": [],
            "window_confidences": [],
        }

        for i, text in tqdm(enumerate(new_text)):
            
            if len(text) < 5 or "sorry" in text.lower():
                result = {
                    "fraction_ai": 50,
                    "fraction_ai_assisted": 50,
                    "fraction_human": 50,
                    "num_ai_segments": 50,
                    "window_labels": [50],
                    "window_ai_assistance_scores": [50],
                    "window_confidences": [50],
                }
            else:
                result = predict_with_backoff(pangram_client, text)

            # if i == 0: import pdb; pdb.set_trace()
            
            if result is None or len(result) == 0:
                # API call failed after all retries
                print(f"failed {i}")
                failed_requests.append(i)
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
                
                final_dict['fraction_ai'].append(fraction_ai)
                final_dict['fraction_ai_assisted'].append(fraction_ai_assisted)
                final_dict['fraction_human'].append(fraction_human)
                final_dict['num_ai_segments'].append(num_ai_segments)
                final_dict['window_labels'].append(window_labels)
                final_dict['window_ai_assistance_scores'].append(window_ai_assistance_scores)
                final_dict['window_confidences'].append(window_confidences)
                                
            except Exception as e:
                print(f"Failed to process result for {i}: {e}")
                failed_requests.append(i)
                continue

        for key in final_dict:
            abstract_dict[key] = final_dict[key]

    # import pdb; pdb.set_trace()

    t_csv = pd.DataFrame(abstract_dict)
        
    t_csv.to_csv(output_csv, index=False)



    