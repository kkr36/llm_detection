"""
Evaluate the *supervised* PN detectors (trained on 2020 text) on the mpe_year
held-out human chunks, and report the average P(AI) per year -- the exact same
figure `eval_pu.py` reports as `mean_ai_prob`, but for the PN models.

Two variants, both trained ONCE on 2020 and applied to all three eval years
(2020/2023/2025), like James:
  1. PN (raw)   : mean over the eval human chunks of P(AI)=softmax(logits)[:,0].
  2. PN (Platt) : same, but each seed's classifier is Platt-scaled on a 2020
                  reference set first (see prepare_temporal.py for the pattern),
                  and P(AI) is read directly off the calibrated probabilities
                  (no second softmax).

The PN checkpoints are the supervised 2020 models used by prepare_temporal.py:
    logging_accuracy_temporal_alpha_full_sentence/sentence_2020/0_<seed>/ArXiv_BERT_3/PN_DistilBert_*.pt
(alpha=0 folders; one PN_*.pt per seed).

Platt calibration set (per seed): the 2020 mpe_year val-split chunks --
AI mirror chunks (positives, true_target=0) + human_abstract chunks
(negatives, true_target=1) -- so the whole mpe_year experiment stays on the same
fronthalf parquets. `fit_platt_scaler` reads those true_targets as its BCE labels,
learning P(human); `PlattCalibratedClassifier` then returns [P(AI), P(human)] so
column 0 is the calibrated P(AI) (matching the raw softmax[:,0] convention).

Needs a GPU. Env: same torch env used for training (llm_embeddings).
Run:
    cd .../PU_learning
    python mpe_year/eval_pn.py
    # options: --pn-root <dir>  --scale-year 2020  --seeds 0 1 2 3 4
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
from helper import initialize_bert_transform, IMDbBERTData, UnlabelData
from platt_scaling import fit_platt_scaler, PlattCalibratedClassifier

from mpe_year.mpe_data import YEARS, read_fronthalf_pu, eval_human_chunks

# The supervised PN models trained on 2020 text (alpha=0 folders under the
# temporal sentence run; one PN_*.pt checkpoint per seed).
DEFAULT_PN_ROOT = os.path.join(
    PU_ROOT, "logging_accuracy_temporal_alpha_full_sentence", "sentence_2020"
)


def pn_checkpoint(pn_root, seed):
    """Path to the single PN_*.pt checkpoint for `seed` (alpha=0 folder)."""
    pat = os.path.join(pn_root, f"0_{seed}", "ArXiv_BERT_3", "PN_DistilBert_*.pt")
    hits = sorted(glob.glob(pat))
    assert len(hits) == 1, f"seed {seed}: expected 1 PN checkpoint, found {hits}"
    return hits[0]


def load_net(model_path, net_type, device):
    net = get_model(net_type)
    state = torch.load(model_path, map_location=device)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    net.to(device)
    return net


def make_u_loader(texts):
    """All texts as a 4-tuple UnlabelData loader (labels are ignored for scoring)."""
    transform = initialize_bert_transform("distilbert-base-uncased")
    ds = IMDbBERTData(texts, [0] * len(texts), transform=transform)
    u = UnlabelData(transform=None, target_transform=None,
                    pos_data=ds.p_data, neg_data=ds.n_data,
                    index=np.arange(len(texts)), data_type="mpe_year")
    return torch.utils.data.DataLoader(u, batch_size=128, shuffle=False)


def make_calib_loader(scale_year, seed):
    """2020 reference calibration set for Platt scaling: AI mirror chunks (P) +
    human chunks (N) from the val split. UnlabelData sets true_target=0 for the
    AI (pos_data) rows and 1 for the human (neg_data) rows, which fit_platt_scaler
    consumes as its labels (so it learns P(human))."""
    # read_fronthalf_pu labels AI mirror chunks 1 and human chunks 0, which is
    # exactly IMDbBERTData's split (label 1 -> p_data, label 0 -> n_data). Encode
    # all texts in ONE transform call so P and N share a padded length.
    texts, labels = read_fronthalf_pu(scale_year, alpha=0.0, split="val", seed=seed)
    transform = initialize_bert_transform("distilbert-base-uncased")
    ds = IMDbBERTData(texts, labels, transform=transform)
    u = UnlabelData(transform=None, target_transform=None,
                    pos_data=ds.p_data, neg_data=ds.n_data,
                    index=np.arange(len(texts)), data_type="mpe_year")
    return torch.utils.data.DataLoader(u, batch_size=128, shuffle=False)


@torch.no_grad()
def mean_p_ai(model, loader, device, already_prob):
    """Average P(AI) over the loader. If `already_prob`, the model already returns
    calibrated probabilities (Platt) so take column 0 directly; otherwise softmax
    the logits and take column 0 (the P(AI)=softmax[:,0] convention)."""
    tot, n = 0.0, 0
    for _, inputs, _, _ in loader:
        inputs = inputs.to(device)
        out = model(inputs)
        p_ai = out[:, 0] if already_prob else torch.softmax(out, dim=-1)[:, 0]
        tot += float(p_ai.sum().cpu())
        n += p_ai.shape[0]
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pn-root", type=str, default=DEFAULT_PN_ROOT)
    ap.add_argument("--net-type", type=str, default="DistilBert")
    ap.add_argument("--scale-year", type=int, default=2020,
                    help="reference year whose val chunks calibrate Platt scaling")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Precompute the shared eval chunks + loaders once per year.
    eval_loaders = {}
    n_eval = {}
    for year in YEARS:
        chunks = eval_human_chunks(year)
        eval_loaders[year] = make_u_loader(chunks)
        n_eval[year] = len(chunks)
        print(f"[{year}] |eval human chunks| = {len(chunks)}")

    rows = []
    for seed in args.seeds:
        ckpt = pn_checkpoint(args.pn_root, seed)
        print(f"\n=== seed {seed}: {ckpt} ===")
        net = load_net(ckpt, args.net_type, device)

        # One Platt scaler per seed, calibrated on the scale-year reference set.
        calib_loader = make_calib_loader(args.scale_year, seed)
        scaler = fit_platt_scaler(model=net, calib_loader=calib_loader, device=device)
        platt_net = PlattCalibratedClassifier(net, scaler).to(device)
        platt_net.eval()

        for year in YEARS:
            raw = mean_p_ai(net, eval_loaders[year], device, already_prob=False)
            cal = mean_p_ai(platt_net, eval_loaders[year], device, already_prob=True)
            rows.append({"method": "pn", "seed": seed, "year": year,
                         "mean_ai_prob": raw, "mean_ai_prob_platt": cal,
                         "n_eval_chunks": n_eval[year], "model_path": ckpt})
            print(f"  [{year}] raw mean P(AI)={raw:.4f}   platt mean P(AI)={cal:.4f}")

    per_model = pd.DataFrame(rows)
    out_dir = os.path.join(PU_ROOT, "mpe_year", "results")
    per_model.to_csv(os.path.join(out_dir, "pn_mpe_per_model.csv"), index=False)

    # Average over seeds per year -> two summary CSVs shaped like pu_mpe.csv, so
    # the plotter can read them the same way (method, year, mean_ai_prob, ...).
    def summarize(value_col, method_name):
        g = (per_model.groupby("year")
             .agg(mean_ai_prob=(value_col, "mean"),
                  mean_ai_prob_std=(value_col, "std"),
                  n_models=(value_col, "count"),
                  n_eval_chunks=("n_eval_chunks", "first"))
             .reset_index())
        g.insert(0, "method", method_name)
        return g

    pn = summarize("mean_ai_prob", "pn")
    pn_platt = summarize("mean_ai_prob_platt", "pn_platt")
    pn.to_csv(os.path.join(out_dir, "pn_mpe.csv"), index=False)
    pn_platt.to_csv(os.path.join(out_dir, "pn_platt_mpe.csv"), index=False)

    print("\nSaved -> pn_mpe.csv, pn_platt_mpe.csv, pn_mpe_per_model.csv")
    print("PN (raw):\n" + pn.to_string(index=False))
    print("PN (platt):\n" + pn_platt.to_string(index=False))


if __name__ == "__main__":
    main()
