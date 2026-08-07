"""Verify the softmax-index convention for the pretrained PN 'all' models.

prepare_heatmap.py:164 asserts these PN models output P(LLM) at softmax index 0.
We confirm empirically: run one model on known-human 2020 abstracts vs known-LLM
(Qwen 2020) abstracts and check which index separates them.
"""
import sys, glob
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_helper import get_model
from data_helper import initialize_bert_transform

DATA = "/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2020_ai_cs._10000_fronthalf_120b_qwen_codex.parquet"
MODEL = glob.glob("/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/logging_accuracy_llm/normal_sentence/alpha_0/all_0/llm_type_all_3/*.pt")[0]

device = "cpu"
net = get_model("DistilBert")
sd = torch.load(MODEL, map_location=device)
sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
net.load_state_dict(sd)
net.eval().to(device)

transform = initialize_bert_transform("distilbert-base-uncased")

df = pd.read_parquet(DATA)
human = df["human_abstract"].dropna().tolist()[:20]
qwen = df[df["Qwen"].notna() & (df["Qwen"] != "")]["Qwen"].tolist()[:20]

def scores(texts):
    x = torch.from_numpy(transform(texts))
    with torch.no_grad():
        out = net(x)
        sm = torch.nn.functional.softmax(out, dim=-1).cpu().numpy()
    return sm  # [:,0], [:,1]

sm_h = scores(human)
sm_l = scores(qwen)
print(f"HUMAN abstracts: mean softmax[:,0]={sm_h[:,0].mean():.3f}  softmax[:,1]={sm_h[:,1].mean():.3f}")
print(f"LLM(Qwen)      : mean softmax[:,0]={sm_l[:,0].mean():.3f}  softmax[:,1]={sm_l[:,1].mean():.3f}")
print()
print("If index 0 = P(LLM): LLM abstracts should have HIGHER softmax[:,0] than human.")
print(f"  human[:,0]={sm_h[:,0].mean():.3f}  <  llm[:,0]={sm_l[:,0].mean():.3f}  ->  {'YES, idx0=P(LLM)' if sm_l[:,0].mean()>sm_h[:,0].mean() else 'NO'}")
