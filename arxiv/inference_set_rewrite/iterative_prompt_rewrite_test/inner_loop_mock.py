### MOCK VERSION of inner_loop.py
### All LLM API calls (query_fns) are replaced with a mock that returns a default abstract.
# must be run using conda env *llm_master*!

import os
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from util import individual_predict, split_into_sentences, clean_text
from strategy import CURRENT_STRATEGY, CURRENT_TIMESTEP
from pathlib import Path
import numpy as np
import sys
import torch
sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_helper import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MOCK_ABSTRACT = "This paper presents a novel approach to the problem at hand. We propose a method that achieves state-of-the-art performance on standard benchmarks. Experimental results demonstrate the effectiveness of the proposed approach. Our method outperforms existing baselines by a significant margin. We conclude with a discussion of limitations and future work."

def mock_strategy(query_fn, orig_abs, llm_label):
    return MOCK_ABSTRACT, None

llm_labels = ["Gemini 3 Preview", "GPT OSS 120b"]

output_csv = f"results_{CURRENT_TIMESTEP}.csv"

if __name__ == "__main__":
    print(f"starting generation for t={CURRENT_TIMESTEP} (MOCK MODE)")
    ### FIXED PARAMS
    subsample_size = 20000
    category = "cs."
    train_year = 2010
    nets = [] # get pre-trained AI detectors
    seeds = 5
    for seed in range(seeds):
        path_to_pretrained = Path(f"/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/logging_accuracy_temporal_alpha_full_sentence/sentence_{train_year}/{0}_{seed}/ArXiv_BERT_3")
        pts = [p for p in path_to_pretrained.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "PN" in p.name]
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

    old_text = []
    new_text = []
    human_text = []

    to_rewrite = 10
    val_start = 2500
    test_start = 2600

    fpath = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{train_year}_ai_{category}_{subsample_size//2}_fronthalf.parquet"
    arxiv_data = pd.read_parquet(fpath)
    llms = [i % len(llm_labels) for i in range(len(arxiv_data))]

    # generate to_rewrite mirrors (mocked)
    print("generating mirrors (MOCKED)")
    for i in tqdm(range(val_start, val_start + to_rewrite)):
        row = arxiv_data.iloc[i]
        orig_abs = row['human_abstract']

        new_abs, _ = mock_strategy(None, orig_abs, llm_labels[llms[i]])

        new_text.append(new_abs)
        human_text.append(orig_abs)

    abstract_dict = {
        "human": [],
        "human_score": [],
        "human_score_avg": [],
        "mirroring_llm": [],
        f"mirror_{CURRENT_TIMESTEP}": [],
        f"mirror_{CURRENT_TIMESTEP}_score": [],
        f"mirror_{CURRENT_TIMESTEP}_score_avg": []
    }

    for i, (human_abstract, mirror_t) in enumerate(list(zip(human_text, new_text))):
        for abstract, label in [(human_abstract, "human"), (mirror_t, f"mirror_{CURRENT_TIMESTEP}")]:
            sentences = clean_text(split_into_sentences(abstract))
            sentence_scores = []
            for sentence in tqdm(sentences):
                avg_score_buffer = []
                for net in nets:
                    _, score = individual_predict(net, device, sentence)
                    avg_score_buffer.append(score)
                sentence_scores.append(np.mean(avg_score_buffer))

            abstract_dict[f"{label}_score"].append(sentence_scores)
            abstract_dict[f"{label}_score_avg"].append(np.mean(sentence_scores))
            abstract_dict[label].append(sentences)
        abstract_dict["mirroring_llm"].append(llm_labels[llms[i]])

    if CURRENT_TIMESTEP != 1:
        t_csv = pd.read_csv(output_csv.replace(str(CURRENT_TIMESTEP), str(CURRENT_TIMESTEP-1)))
        t_csv[f"mirror_{CURRENT_TIMESTEP}"] = abstract_dict[f"mirror_{CURRENT_TIMESTEP}"]
        t_csv[f"mirror_{CURRENT_TIMESTEP}_score"] = abstract_dict[f"mirror_{CURRENT_TIMESTEP}_score"]
        t_csv[f"mirror_{CURRENT_TIMESTEP}_score_avg"] = abstract_dict[f"mirror_{CURRENT_TIMESTEP}_score_avg"]
    else:
        t_csv = pd.DataFrame(abstract_dict)

    t_csv.to_csv(output_csv, index=False)
