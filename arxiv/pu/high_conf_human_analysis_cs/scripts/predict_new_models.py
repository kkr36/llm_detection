"""
Score every sentence of the 4 data_raw_{year}_{set}.csv files with the NEW
2025 TEDn DistilBERT detectors (5 seeds) in /share/garg/arxiv_kaggle/2025_models.

Convention (verified in train_2025_tedn.py + estimator.p_probs): the positive class
is the LLM mirror, and p_probs uses softmax(logits)[:, 0], so
    P(LLM)   = softmax(logits, dim=-1)[:, 0]
    P(human) = 1 - P(LLM) = softmax(...)[:, 1]
(The nets were saved under torch.nn.DataParallel, so keys carry a 'module.' prefix.)

Writes ../data/preds_{year}_{set}.csv  (sentence, abstract_id, p_llm_m0..m4)
for (2020,2025) x (train,val). Assembly of the corpus/subset CSVs is done
separately on CPU (assemble_corpora.py).
"""
import os, sys, glob
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_helper import get_model
from data_helper import initialize_bert_transform

DATA = "/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis_cs/data"
MODEL_GLOB = "/share/garg/arxiv_kaggle/2025_models/ArXiv2025_backhalf_3/*.pt"
COMBOS = [("2020", "train"), ("2020", "val"), ("2025", "train"), ("2025", "val")]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[predict] device={device}")

model_paths = sorted(glob.glob(MODEL_GLOB))
assert len(model_paths) == 5, f"expected 5 new models, found {len(model_paths)}: {model_paths}"
print("[predict] new models:")
for p in model_paths:
    print("   ", os.path.basename(p))

nets = []
for p in model_paths:
    net = get_model("DistilBert")
    sd = torch.load(p, map_location=device)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval().to(device)
    nets.append(net)

transform = initialize_bert_transform("distilbert-base-uncased")


@torch.no_grad()
def p_llm(net, texts, batch=64):
    out = np.empty(len(texts), dtype=np.float64)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        x = torch.from_numpy(transform(chunk)).to(device)
        logits = net(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, 0]
        out[i:i + batch] = probs.detach().cpu().numpy().ravel()
    return out


# ---- sanity: on 2020 fronthalf parquet, human should score lower P(LLM) than an LLM mirror
_df = pd.read_parquet("/share/garg/arxiv_kaggle/multillm/data_raw/"
                      "arxiv_2020_ai_cs._10000_fronthalf.parquet")
_hum = _df["human_abstract"].dropna().tolist()[:40]
_llm = _df[_df["Llama 3.3 70b Instruct"].notna()
           & (_df["Llama 3.3 70b Instruct"] != "")]["Llama 3.3 70b Instruct"].tolist()[:40]
_h, _l = p_llm(nets[0], _hum).mean(), p_llm(nets[0], _llm).mean()
print(f"[sanity] new-model0 mean P(LLM): human={_h:.3f}  llm={_l:.3f}  "
      f"({'OK idx0=P(LLM)' if _l > _h else 'WARNING'})")
assert _l > _h, "softmax[:,0] does not behave as P(LLM) for new models; check convention!"

for year, st in COMBOS:
    df = pd.read_csv(f"{DATA}/data_raw_{year}_{st}.csv")
    sents = df["sentence"].astype(str).tolist()
    preds = np.zeros((len(sents), len(nets)))
    for mi, net in enumerate(nets):
        preds[:, mi] = p_llm(net, sents)
        print(f"[{year}_{st}] scored model {mi}")
    out = df[["sentence", "abstract_id"]].copy()
    for mi in range(len(nets)):
        out[f"p_llm_m{mi}"] = preds[:, mi]
    out.to_csv(f"{DATA}/preds_{year}_{st}.csv", index=False)
    pm = preds.mean(axis=1)
    print(f"[{year}_{st}] wrote preds_{year}_{st}.csv  ({len(out)} rows) "
          f"mean P(LLM)={pm.mean():.3f}  frac P(human)>0.9={(1-pm > 0.9).mean():.3f}")

print("[predict] DONE")
