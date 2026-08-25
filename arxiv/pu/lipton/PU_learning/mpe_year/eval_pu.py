"""
Evaluate the trained PU/TEDn detectors (from mpe_year/train_pu.py) on the held-out
2,000 human abstracts per year, and report the estimated AI fraction (MPE).

For each (year, seed model):
    P = confirmed AI positives (val-split mirror 75-word chunks)
    U = the held-out human_abstract 75-word chunks (unknown mixture) -- the same
        chunks Pangram and James score
    -> BBE_estimator(P_probs, U_probs) gives the mixture-proportion estimate
       (fraction of the eval human chunks the detector attributes to AI).
Also reports the plug-in mean P(AI)=softmax[:,0] over U for reference.
Results are averaged over whatever seeds are found and written to results/pu_mpe.csv.

Needs a GPU. Env: same torch env used for training.
Run:
    cd .../PU_learning
    python mpe_year/eval_pu.py --log-dir /share/garg/arxiv_kaggle/mpe_year_models
"""

import os
import sys
import glob
import argparse

PU_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PU_ROOT not in sys.path:
    sys.path.insert(0, PU_ROOT)

import numpy as np
import pandas as pd
import torch

from model_helper import get_model
from helper import initialize_bert_transform, IMDbBERTData, PosData, UnlabelData
from estimator import p_probs, u_probs, BBE_estimator

from mpe_year.mpe_data import YEARS, read_fronthalf_pu, eval_human_chunks

DEFAULT_LOG_DIR = "/share/garg/arxiv_kaggle/mpe_year_models"


def make_p_loader(texts):
    transform = initialize_bert_transform("distilbert-base-uncased")
    ds = IMDbBERTData(texts, [1] * len(texts), transform=transform)
    p = PosData(transform=None, target_transform=None,
                data=ds.p_data, index=np.arange(len(ds.p_data)), data_type="mpe_year")
    return torch.utils.data.DataLoader(p, batch_size=128, shuffle=False)


def make_u_loader(texts):
    """All texts as unlabeled with dummy label 0 (BBE's estimate ignores U targets)."""
    transform = initialize_bert_transform("distilbert-base-uncased")
    ds = IMDbBERTData(texts, [0] * len(texts), transform=transform)
    u = UnlabelData(transform=None, target_transform=None,
                    pos_data=ds.p_data, neg_data=ds.n_data,
                    index=np.arange(len(texts)), data_type="mpe_year")
    return torch.utils.data.DataLoader(u, batch_size=128, shuffle=False)


def load_net(model_path, net_type, device):
    net = get_model(net_type)
    state = torch.load(model_path, map_location=device)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    net.to(device)
    return net


def run_year(year, log_dir, net_type, seed_for_p, device):
    model_paths = sorted(glob.glob(os.path.join(log_dir, str(year), "*.pt")))
    if not model_paths:
        print(f"[{year}] no models in {os.path.join(log_dir, str(year))}; skipping")
        return []

    # confirmed-positive P-set: AI mirror chunks from the val split
    p_texts, p_labels = read_fronthalf_pu(year, alpha=0.0, split="val", seed=seed_for_p)
    p_texts = [t for t, l in zip(p_texts, p_labels) if l == 1]
    u_texts = eval_human_chunks(year)  # shared 75-word eval chunks
    print(f"[{year}] |P|={len(p_texts)}  |U (eval human chunks)|={len(u_texts)}")

    p_loader = make_p_loader(p_texts)
    u_loader = make_u_loader(u_texts)

    rows = []
    for mp in model_paths:
        net = load_net(mp, net_type, device)
        pos_probs = p_probs(net, device, p_loader)
        unlabeled_probs, unlabeled_targets = u_probs(net, device, u_loader)
        mpe, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
        mean_ai_prob = float(np.mean(unlabeled_probs[:, 0]))  # P(LLM)=softmax[:,0]
        rows.append({
            "method": "pu",
            "year": year,
            "mpe": float(mpe),
            "mean_ai_prob": mean_ai_prob,
            "n_eval_chunks": len(u_texts),
            "n_p_chunks": len(p_texts),
            "model_path": mp,
        })
        print(f"  {os.path.basename(mp)}: MPE={mpe:.3f}  mean_ai_prob={mean_ai_prob:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    ap.add_argument("--net-type", type=str, default="DistilBert")
    ap.add_argument("--seed-for-p", type=int, default=0, help="seed for the P-set (val mirror) sampling")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = []
    for year in YEARS:
        all_rows += run_year(year, args.log_dir, args.net_type, args.seed_for_p, device)

    if not all_rows:
        print("No PU models found -- train first with mpe_year/train_pu.py.")
        return

    per_model = pd.DataFrame(all_rows)
    per_model.to_csv(os.path.join(PU_ROOT, "mpe_year", "results", "pu_mpe_per_model.csv"), index=False)

    # average over seeds per year
    summary = (per_model.groupby("year")
               .agg(mpe=("mpe", "mean"), mpe_std=("mpe", "std"),
                    mean_ai_prob=("mean_ai_prob", "mean"),
                    n_models=("mpe", "count"),
                    n_eval_chunks=("n_eval_chunks", "first"))
               .reset_index())
    summary.insert(0, "method", "pu")
    out = os.path.join(PU_ROOT, "mpe_year", "results", "pu_mpe.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
