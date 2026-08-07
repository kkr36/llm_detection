"""
Temporal human-writing analysis: run the 5 pretrained PN 'all' detectors on
human-submitted arXiv abstracts from 2020 (pre-LLM, treated as ground-truth human)
and 2025 (unknown authorship).

Convention (verified against prepare_heatmap.py:164): for these PN models,
    P(LLM) = softmax(logits, dim=-1)[:, 0]
    P(human) = 1 - P(LLM)

Outputs (written to ../data):
  2020_human_sample.csv          100 sampled 2020 human abstracts + preds
  2025_abstract_predictions.csv  all 500 sampled 2025 abstracts + abstract-level preds
  2025_sentence_predictions.csv  every sentence of every 2025 abstract + per-model P(LLM)
  2025_pseudolabel.csv           100 abstracts with mean P(human) > 0.9
  2020_sentence_predictions.csv  every sentence of every 2020 abstract + per-model P(LLM)
Each abstract row carries a `prefix_key` used later to join arXiv categories.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_helper import get_model
from data_helper import initialize_bert_transform, split_into_sentences

DATA_DIR = "/share/garg/arxiv_kaggle/multillm/data_raw"
P2020 = f"{DATA_DIR}/arxiv_2020_ai_cs._10000_fronthalf_120b_qwen_codex.parquet"
P2025 = f"{DATA_DIR}/arxiv_2025_ai_cs._10000_fronthalf.parquet"
MODEL_GLOB = ("/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/"
              "logging_accuracy_llm/normal_sentence/alpha_0/all_*/llm_type_all_3/*.pt")
OUT = "/home/kkr36/llm_detection/arxiv/temporal_human_analysis/data"

N_2020 = 100      # sampled genuine-human abstracts (all used, all scored)
N_2025 = 800      # sampled 2025 abstracts to label (need >=100 with P(human)>0.9 for pseudolabels)
SEED = 42

os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[pipeline] device={device}")


def prefix_key(text, n=100):
    """Whitespace-normalized first n chars; stable join key to full arXiv abstract."""
    return " ".join(str(text).split())[:n]


# ---------------------------------------------------------------- load models
model_paths = sorted(glob.glob(MODEL_GLOB))
assert len(model_paths) == 5, f"expected 5 models, found {len(model_paths)}: {model_paths}"
print("[pipeline] models:")
for p in model_paths:
    print("   ", p)

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
    """Return P(LLM)=softmax[:,0] for a list of texts."""
    out = np.empty(len(texts), dtype=np.float64)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        x = torch.from_numpy(transform(chunk)).to(device)
        logits = net(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, 0]
        out[i:i + batch] = probs.detach().cpu().numpy().ravel()
    return out


# ------------------------------------------------- sanity check on convention
df2020_full = pd.read_parquet(P2020)
_hum = df2020_full["human_abstract"].dropna().tolist()[:30]
_qwen = df2020_full[df2020_full["Qwen"].notna() & (df2020_full["Qwen"] != "")]["Qwen"].tolist()[:30]
_h = p_llm(nets[0], _hum).mean()
_l = p_llm(nets[0], _qwen).mean()
print(f"[sanity] model0 mean P(LLM): human2020={_h:.3f}  qwen2020={_l:.3f}  "
      f"({'OK idx0=P(LLM)' if _l > _h else 'WARNING convention'})")
assert _l > _h, "softmax[:,0] does not behave as P(LLM); check convention!"


# --------------------------------------------------------------- sample data
df2025_full = pd.read_parquet(P2025)

s2020 = (df2020_full[df2020_full["human_abstract"].notna()]
         .sample(n=N_2020, random_state=SEED).reset_index(drop=True))
s2025 = (df2025_full[df2025_full["human_abstract"].notna()]
         .sample(n=N_2025, random_state=SEED).reset_index(drop=True))
print(f"[pipeline] sampled 2020={len(s2020)}  2025={len(s2025)}")


def score_year(df, tag):
    """Sentence-split each abstract, score every sentence with every model,
    return (abstract_df, sentence_df)."""
    abstracts = df["human_abstract"].tolist()
    # sentence-split, keeping abstract membership
    sents, abs_idx = [], []
    for ai, ab in enumerate(abstracts):
        ss, _ = split_into_sentences([ab], [0])
        ss = [s for s in ss if s.strip()]
        if not ss:
            ss = [ab.strip()]
        sents.extend(ss)
        abs_idx.extend([ai] * len(ss))
    abs_idx = np.array(abs_idx)
    print(f"[{tag}] {len(abstracts)} abstracts -> {len(sents)} sentences")

    # per-model P(LLM) for every sentence
    sent_pllm = np.zeros((len(sents), len(nets)), dtype=np.float64)
    for mi, net in enumerate(nets):
        sent_pllm[:, mi] = p_llm(net, sents)
        print(f"[{tag}] scored model {mi}")

    # sentence-level dataframe
    sent_df = pd.DataFrame({
        "abstract_id": [f"{tag}_{i}" for i in abs_idx],
        "abstract_local_idx": abs_idx,
        "sentence_idx": [j for i in range(len(abstracts)) for j in range(int((abs_idx == i).sum()))],
        "sentence": sents,
    })
    for mi in range(len(nets)):
        sent_df[f"p_llm_m{mi}"] = sent_pllm[:, mi]
    sent_df["p_llm_mean_over_models"] = sent_pllm.mean(axis=1)

    # abstract-level aggregation: per model, mean over its sentences -> then over models
    rows = []
    for ai, ab in enumerate(abstracts):
        mask = abs_idx == ai
        per_model_pllm = sent_pllm[mask].mean(axis=0)          # mean P(LLM) over sentences, per model
        per_model_phuman = 1.0 - per_model_pllm
        rows.append({
            "abstract_id": f"{tag}_{ai}",
            "abstract_local_idx": ai,
            "human_abstract": ab,
            "prefix_key": prefix_key(ab),
            "n_sentences": int(mask.sum()),
            **{f"p_llm_m{mi}": per_model_pllm[mi] for mi in range(len(nets))},
            "p_llm_mean_over_models": float(per_model_pllm.mean()),
            "p_human_mean_over_models": float(per_model_phuman.mean()),
            "p_llm_std_over_models": float(per_model_pllm.std()),
        })
    abs_df = pd.DataFrame(rows)
    return abs_df, sent_df


abs2020, sent2020 = score_year(s2020, "2020")
abs2025, sent2025 = score_year(s2025, "2025")

# ------------------------------------------------------------------- save
abs2020.to_csv(f"{OUT}/2020_human_sample.csv", index=False)
sent2020.to_csv(f"{OUT}/2020_sentence_predictions.csv", index=False)
abs2025.to_csv(f"{OUT}/2025_abstract_predictions.csv", index=False)
sent2025.to_csv(f"{OUT}/2025_sentence_predictions.csv", index=False)

# pseudo-labelled high-confidence human 2025 abstracts
high_human = abs2025[abs2025["p_human_mean_over_models"] > 0.9].reset_index(drop=True)
print(f"[pipeline] 2025 abstracts with mean P(human)>0.9: {len(high_human)} / {len(abs2025)}")
n_take = min(100, len(high_human))
pseudo = high_human.sample(n=n_take, random_state=SEED).reset_index(drop=True)
pseudo.to_csv(f"{OUT}/2025_pseudolabel.csv", index=False)
print(f"[pipeline] wrote 2025_pseudolabel.csv with {len(pseudo)} abstracts")

print("[pipeline] DONE")
