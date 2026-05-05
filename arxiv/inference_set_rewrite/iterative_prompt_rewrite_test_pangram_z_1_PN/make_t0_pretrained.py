# must be run using conda env *llm_master*!
# Takes a t=0 CSV (with an "original" column) and scores all texts through the
# pre-trained models. Outputs columns:
#   original, mirror_0, original_score, mirror_0_score,
#   original_score_avg, mirror_0_score_avg
# where original == mirror_0 and all score columns are duplicated accordingly.

import argparse
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_helper import get_model
from util import individual_predict, split_into_sentences, clean_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("input_csv", help="Path to t=0 CSV file (must contain an 'original' column)")
parser.add_argument("--output_csv", default=None, help="Output path (default: <input>_pretrained.csv)")
args = parser.parse_args()

input_path = Path(args.input_csv)
output_path = Path(args.output_csv) if args.output_csv else input_path.with_name(input_path.stem + "_pretrained.csv")

input_data = pd.read_csv(input_path)
assert "original" in input_data.columns, "'original' column not found in input CSV"

# Load pre-trained models
nets = []
seeds = 5
for seed in range(seeds):
    path_to_pretrained = Path(
        f"/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/logging_accuracy_xy/normal_sentence/alpha_0/{seed}/xz/xy_3"
    )
    pts = [p for p in path_to_pretrained.iterdir() if p.is_file() and p.name.lower().endswith(".pt") and "PN" in p.name]
    assert len(pts) == 1, f"Expected 1 .pt file for seed {seed}, found {len(pts)}"
    net = get_model("DistilBert")
    state_dict = torch.load(pts[0], map_location=device)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    nets.append(net)

print("Loaded models")

texts = input_data["original"].tolist()
scores = []
scores_avg = []

for text in tqdm(texts):
    sentences = clean_text(split_into_sentences(text))
    sentence_scores = []
    for sentence in tqdm(sentences, leave=False):
        buf = []
        for net in nets:
            _, score = individual_predict(net, device, sentence)
            # score = 1 - score  # PU correction
            buf.append(score.cpu())
        sentence_scores.append(np.mean(buf))
    scores.append(sentence_scores)
    scores_avg.append(np.mean(sentence_scores))

out = pd.DataFrame({
    "original": texts,
    "mirror_0": texts,
    "original_score": scores,
    "mirror_0_score": scores,
    "original_score_avg": scores_avg,
    "mirror_0_score_avg": scores_avg,
})

out.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
