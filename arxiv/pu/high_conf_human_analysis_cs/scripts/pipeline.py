"""
Stage 1 -- sample + inference (cs.AI only).

Sample human-submitted arXiv abstracts, restricted to the **cs.AI** subcategory,
directly from the raw metadata snapshot (NOT from the existing rewrite parquets),
excluding any abstract already present in the two front-half parquets. Score every
sentence with the 5 pretrained PN "all" DistilBERT detectors.

Year convention (matches the project's existing `subsample_by_year.py`):
    year = update_date.split("-")[0]
`update_date` is the date of the paper's most recent revision, so update_date==2020
guarantees the current abstract has not been touched since 2020 -> genuinely
pre-ChatGPT / ground-truth human. 2025 abstracts have unknown authorship.

cs.AI filter: 'cs.AI' present as a whitespace-delimited token in `categories`.

Exclusion: an abstract is skipped if its whitespace-normalized 100-char prefix
matches the prefix of any `human_abstract` (front-half) in either
`arxiv_2020_ai_cs._10000_fronthalf.parquet` or
`arxiv_2025_ai_cs._10000_fronthalf.parquet`. (The parquet abstract is the front
half, so its first 100 chars equal the first 100 chars of the full abstract.)

Detector convention (verified against prepare_heatmap.py:164 and empirically):
    P(LLM)   = softmax(logits, dim=-1)[:, 0]
    P(human) = 1 - P(LLM)

Sampling (seed 42):
  2020: 200 human abstracts (first 100 -> _train, last 100 -> _val)
  2025: 1000 abstracts       (first 500 -> _train, last 500 -> _val)

Per (year, set) outputs (../data):
  data_raw_{year}_{set}.csv        2 cols: sentence, abstract_id
  all_predictions_{year}_{set}.csv 7 cols: sentence, abstract_id, p_llm_m0..m4
  high_conf_human_{year}_{set}.csv 2 cols: sentence, abstract_id
                                   (subset of data_raw whose mean P(human) over the
                                    5 models > 0.9)
Also writes (for downstream analysis, not part of the required deliverable):
  sampled_abstracts_{year}.csv     sampled abstracts + arxiv id + categories + set
  sentence_predictions_all.csv     every sentence, both years/sets, all preds + meta
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")
from model_helper import get_model
from data_helper import initialize_bert_transform, split_into_sentences

META = "/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json"
DATA_DIR = "/share/garg/arxiv_kaggle/multillm/data_raw"
EXCLUDE_PARQUETS = [
    f"{DATA_DIR}/arxiv_2020_ai_cs._10000_fronthalf.parquet",
    f"{DATA_DIR}/arxiv_2025_ai_cs._10000_fronthalf.parquet",
]
MODEL_GLOB = ("/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/"
              "logging_accuracy_llm/normal_sentence/alpha_0/all_*/llm_type_all_3/*.pt")
OUT = "/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis_cs/data"

CATEGORY_TOKEN = "cs.AI"
YEARS = {"2020": 200, "2025": 1000}          # year -> number of abstracts to sample
SPLIT = {"2020": 100, "2025": 500}           # first N -> train, remaining -> val
SEED = 42
os.makedirs(OUT, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[pipeline] device={device}")


def prefix_key(text, n=100):
    """Whitespace-normalized first n chars; stable identity key for an abstract."""
    return " ".join(str(text).split())[:n]


# --------------------------------------------------------- exclusion prefix set
exclude_keys = set()
for pq in EXCLUDE_PARQUETS:
    col = pd.read_parquet(pq, columns=["human_abstract"])["human_abstract"]
    exclude_keys.update(prefix_key(a) for a in col.dropna().tolist())
print(f"[pipeline] built exclusion set of {len(exclude_keys)} abstract prefixes "
      f"from {len(EXCLUDE_PARQUETS)} parquets")


# ------------------------------------------------- stream snapshot, collect pool
def has_cs_ai(categories):
    return CATEGORY_TOKEN in (categories or "").split()

pool = {y: [] for y in YEARS}       # year -> list of dict(abstract, arxiv_id, categories, prefix_key)
seen_prefix = {y: set() for y in YEARS}
n_lines = n_excluded = n_dup = 0
with open(META, "r") as fh:
    for line in fh:
        n_lines += 1
        if n_lines % 1_000_000 == 0:
            print(f"[pipeline] scanned {n_lines:,} records; "
                  f"pool sizes " + ", ".join(f"{y}={len(pool[y])}" for y in YEARS))
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not has_cs_ai(rec.get("categories")):
            continue
        ab = rec.get("abstract")
        if not ab or not ab.strip():
            continue
        yr = (rec.get("update_date") or "").split("-")[0]
        if yr not in pool:
            continue
        k = prefix_key(ab)
        if k in exclude_keys:
            n_excluded += 1
            continue
        if k in seen_prefix[yr]:          # de-dup within a year
            n_dup += 1
            continue
        seen_prefix[yr].add(k)
        pool[yr].append({
            "abstract": ab.strip(),
            "arxiv_id": rec.get("id", ""),
            "categories": rec.get("categories", ""),
            "primary_category": (rec.get("categories", "").split() or [""])[0],
            "update_date": rec.get("update_date", ""),
            "prefix_key": k,
        })
print(f"[pipeline] done streaming {n_lines:,} records. "
      f"cs.AI pool: " + ", ".join(f"{y}={len(pool[y])}" for y in YEARS) +
      f"  (excluded={n_excluded}, dup={n_dup})")
for y, n in YEARS.items():
    assert len(pool[y]) >= n, f"only {len(pool[y])} cs.AI {y} abstracts, need {n}"


# --------------------------------------------------------------- load 5 models
model_paths = sorted(glob.glob(MODEL_GLOB))
assert len(model_paths) == 5, f"expected 5 models, found {len(model_paths)}"
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
    out = np.empty(len(texts), dtype=np.float64)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        x = torch.from_numpy(transform(chunk)).to(device)
        logits = net(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, 0]
        out[i:i + batch] = probs.detach().cpu().numpy().ravel()
    return out


# ------------------------------------------------- sanity check the convention
_df20 = pd.read_parquet(EXCLUDE_PARQUETS[0])
_hum = _df20["human_abstract"].dropna().tolist()[:30]
_llm = _df20[_df20["Llama 3.3 70b Instruct"].notna()
             & (_df20["Llama 3.3 70b Instruct"] != "")]["Llama 3.3 70b Instruct"].tolist()[:30]
_h, _l = p_llm(nets[0], _hum).mean(), p_llm(nets[0], _llm).mean()
print(f"[sanity] model0 mean P(LLM): human={_h:.3f}  llm={_l:.3f}  "
      f"({'OK idx0=P(LLM)' if _l > _h else 'WARNING'})")
assert _l > _h, "softmax[:,0] does not behave as P(LLM); check convention!"


# --------------------------------------------------------------- sample + set
rng = np.random.default_rng(SEED)
sampled = {}
for y, n in YEARS.items():
    idx = rng.choice(len(pool[y]), size=n, replace=False)
    rows = [pool[y][i] for i in idx]
    for j, r in enumerate(rows):
        r["set"] = "train" if j < SPLIT[y] else "val"
        r["order"] = j
    sampled[y] = rows
    n_tr = sum(r["set"] == "train" for r in rows)
    print(f"[pipeline] {y}: sampled {len(rows)}  (train={n_tr}, val={len(rows)-n_tr})")


# ----------------------------------------------------- score + write per set
NMODELS = len(nets)
all_sent_frames = []
for y in YEARS:
    rows = sampled[y]
    # persist the abstract-level sample (for the analysis stage)
    pd.DataFrame(rows)[["arxiv_id", "prefix_key", "primary_category", "categories",
                        "update_date", "set", "order", "abstract"]] \
        .to_csv(f"{OUT}/sampled_abstracts_{y}.csv", index=False)

    for st in ("train", "val"):
        subset = [r for r in rows if r["set"] == st]
        sents, abs_ids = [], []
        for r in subset:
            ss, _ = split_into_sentences([r["abstract"]], [0])
            ss = [s for s in ss if s.strip()]
            if not ss:
                ss = [r["abstract"].strip()]
            sents.extend(ss)
            abs_ids.extend([r["arxiv_id"]] * len(ss))
        print(f"[{y}_{st}] {len(subset)} abstracts -> {len(sents)} sentences")

        preds = np.zeros((len(sents), NMODELS))
        for mi, net in enumerate(nets):
            preds[:, mi] = p_llm(net, sents)
            print(f"[{y}_{st}] scored model {mi}")
        p_human_mean = 1.0 - preds.mean(axis=1)

        base = pd.DataFrame({"sentence": sents, "abstract_id": abs_ids})
        # data_raw: 2 columns
        base.to_csv(f"{OUT}/data_raw_{y}_{st}.csv", index=False)
        # all_predictions: 7 columns
        allp = base.copy()
        for mi in range(NMODELS):
            allp[f"p_llm_m{mi}"] = preds[:, mi]
        allp.to_csv(f"{OUT}/all_predictions_{y}_{st}.csv", index=False)
        # high_conf_human: subset of data_raw where mean P(human) > 0.9
        hc = base[p_human_mean > 0.9].reset_index(drop=True)
        hc.to_csv(f"{OUT}/high_conf_human_{y}_{st}.csv", index=False)
        print(f"[{y}_{st}] wrote data_raw({len(base)}), all_predictions({len(allp)}), "
              f"high_conf_human({len(hc)}) [{len(hc)/max(len(base),1):.1%} of sentences]")

        # accumulate for combined analysis file
        f = allp.copy()
        f["year"] = int(y)
        f["set"] = st
        f["p_llm_mean_over_models"] = preds.mean(axis=1)
        f["p_human_mean_over_models"] = p_human_mean
        all_sent_frames.append(f)

sent_all = pd.concat(all_sent_frames, ignore_index=True)
sent_all.to_csv(f"{OUT}/sentence_predictions_all.csv", index=False)
print(f"[pipeline] sentence_predictions_all.csv: {len(sent_all)} sentences")
print("[pipeline] DONE")
