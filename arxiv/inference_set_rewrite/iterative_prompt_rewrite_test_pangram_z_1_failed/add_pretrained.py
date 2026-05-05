### DO NOT OVERWRITE, EDIT, OR TOUCH ANYTHING IN THIS FILE ###
# must be run using conda env *llm_master*!

import os
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from openai_api import openai_oss_query
from gemini_api import call_gemini_3
from util import individual_predict, split_into_sentences, clean_text, predict_with_backoff
from pathlib import Path
import numpy as np
import sys
import torch
sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_helper import get_model
from pangram import Pangram

with open("/home/kkr36/creds.json", 'r') as handle:
    pangram_api_key = json.load(handle)['pangram_api_key']
pangram_client = Pangram(api_key=pangram_api_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

split = "val"
pangram = False
to_mirror = 15
timestep = 10
input_csv = f"results_{timestep}_oss_{split}_{to_mirror}"
input_data = pd.read_csv(f"{input_csv}.csv")

if __name__ == "__main__":
    ### FIXED PARAMS

    train_year = 2010
    nets = [] # get pre-trained AI detectors; they take in text (a sentence) and output P(LLM) = 1 - P(original); we want to minimize this with our LLM mirrors; i.e., fool the detector
    seeds = 5
    for seed in range(seeds):
        path_to_pretrained = Path(f"/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/logging_accuracy_xy/normal_sentence/alpha_0.25/{seed}/xz/xy_3")
        pts = [p for p in path_to_pretrained.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "TEDn" in p.name]
        assert(len(pts) == 1)

        model_path = pts[0]

        net = get_model("DistilBert")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.eval()
        net.to(device)
        nets.append(net)
    
    print("loaded models")
    
    # pass through pre-trained classification model -- higher numbers = lower P(original); we want to MINIMIZE the scores of this classifier
    # specifically, the average score over new_text should be lower than the average score over old_text (meaning CURRENT_STRATEGY is better than the original), and approach average score over original abstracts
    # each row of abstract_dict represents sentences of one original abstract, the sentences of its llm mirror, and scores per sentence
    abstract_dict = {
        "original_score": [],
        "original_score_avg": [],
        f"mirror_{timestep}_score": [],
        f"mirror_{timestep}_score_avg": []
    }

    original_text, new_text = input_data["original"].tolist(), input_data[f"mirror_{timestep}"].tolist()
    
    for i, (original_abstract, mirror_t) in enumerate(list(zip(original_text, new_text))):
        for abstract, label in [(original_abstract, "original"), (mirror_t, f"mirror_{timestep}")]:
            sentences = clean_text(split_into_sentences(abstract))
            sentence_scores = []
            for sentence in tqdm(sentences):
                avg_score_buffer = []
                for net in nets:
                    _, score = individual_predict(net, device, sentence)
                    score = 1-score # for PU only
                    avg_score_buffer.append(score.cpu())
                sentence_scores.append(np.mean(avg_score_buffer))

            # update dict with sentences, scores
            abstract_dict[f"{label}_score"].append(sentence_scores)
            abstract_dict[f"{label}_score_avg"].append(np.mean(sentence_scores))

    if pangram:
        failed_requests = []
        final_dict = {
            "fraction_ai": [],
            "fraction_ai_assisted": [],
            "fraction_original": [],
            "num_ai_segments": [],
            "window_labels": [],
            "window_ai_assistance_scores": [],
            "window_confidences": [],
        }

        for i, text in tqdm(enumerate(new_text)):
            
            if not isinstance(text, str) or len(text) < 5 or "sorry" in text.lower():
                result = {
                    "fraction_ai": 50,
                    "fraction_ai_assisted": 50,
                    "fraction_original": 50,
                    "num_ai_segments": 50,
                    "window_labels": [50],
                    "window_ai_assistance_scores": [50],
                    "window_confidences": [50],
                    "text": "BAD"
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
                fraction_original = result.get('fraction_original', np.nan)
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
                final_dict['fraction_original'].append(fraction_original)
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

    # t_csv = pd.DataFrame(abstract_dict)
    for key in abstract_dict:
        input_data[key] = abstract_dict[key]

    input_data.to_csv(f"{input_csv}_pretrained.csv", index=False)



    