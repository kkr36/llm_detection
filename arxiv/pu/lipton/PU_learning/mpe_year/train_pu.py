"""
Train a PU/TEDn detector per year for mpe_year, on the first 8,000 rows of
arxiv_{year}_ai_cs._10000_fronthalf.parquet.

  labeled positives (P) = AI mirrors
  unlabeled        (U) = human_abstract text (+ optional alpha-fraction mirrors)

This is a thin generalization of scripts/train_2025/train_2025_tedn.py to an
arbitrary year + the plain _fronthalf parquet, using mpe_data.read_fronthalf_pu.
All training building blocks are imported from the existing modules; nothing here
is modified in place.

Needs a GPU (DistilBert). Env: any torch env with transformers (e.g. llm_embeddings
/ pu_env). Run one model:
    cd .../PU_learning
    python mpe_year/train_pu.py --year 2020 --seed 0
Or use mpe_year/run_pu_train.sbatch for the (year x seed) array.
Models are written to  {log_dir}/{year}/  and consumed by mpe_year/eval_pu.py.
"""

import os
import sys
import time
import random
import argparse

PU_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PU_ROOT not in sys.path:
    sys.path.insert(0, PU_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score

from algorithm import *      # train_PU_discard, rank_inputs, validate
from model_helper import *   # get_model
from helper import *         # IMDbBERTData, get_PUDataSplits1, clean_text, shuffle, initialize_bert_transform
from estimator import *      # p_probs, u_probs, BBE_estimator
from baselines import *      # scott_estimator, estimator_CM_EN

from mpe_year.mpe_data import read_fronthalf_pu

np.set_printoptions(suppress=True, precision=1)

DEFAULT_LOG_DIR = "/share/garg/arxiv_kaggle/mpe_year_models"


def build_loaders(year, alpha, seed, clean):
    transform = initialize_bert_transform("distilbert-base-uncased")

    train_texts, train_labels = read_fronthalf_pu(year, alpha, "train", seed)
    val_texts, val_labels = read_fronthalf_pu(year, alpha, "val", seed)

    if clean:
        train_texts = clean_text(train_texts)
        val_texts = clean_text(val_texts)

    train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
    val_dataset = IMDbBERTData(val_texts, val_labels, transform=transform)

    p_traindata, u_traindata = get_PUDataSplits1(train_dataset, data_type="mpe_year")
    p_validdata, u_validdata = get_PUDataSplits1(val_dataset, data_type="mpe_year")

    p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=8, shuffle=shuffle)
    u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=8, shuffle=shuffle)
    p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=128, shuffle=shuffle)
    u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, shuffle=shuffle)
    return (p_trainloader, u_trainloader, p_validloader, u_validloader, u_traindata)


def main():
    ap = argparse.ArgumentParser(description="Train PU/TEDn on arxiv_{year}_..._fronthalf (mirrors=P, humans=U)")
    ap.add_argument("--year", type=int, required=True, choices=[2020, 2023, 2025])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="injected-mirror fraction of U; 0 = U is purely human_abstract text")
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.00001)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--clean", default=False, action="store_true",
                    help="strip non-typable chars (run the sbatch with --clean, matching train_2025)")
    ap.add_argument("--net-type", type=str, default="DistilBert")
    ap.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    args = ap.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = os.path.join(args.log_dir, str(args.year))
    os.makedirs(log_dir, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")

    (p_trainloader, u_trainloader, p_validloader, u_validloader, u_traindata) = build_loaders(
        args.year, args.alpha, args.seed, args.clean)
    train_unlabeled_size = len(u_traindata.targets)

    net = get_model(args.net_type).to(device)
    if device.startswith("cuda"):
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(net.parameters(), lr=args.lr)

    # ---- TEDn training loop (CVIR/TEDn branch of train_PU_one_year.py / train_2025_tedn.py) ----
    alpha_estimate = 0.0
    for epoch in range(args.epochs):
        keep_samples, _ = rank_inputs(epoch, net, u_trainloader, device, alpha_estimate,
                                      u_size=train_unlabeled_size)
        train_acc = train_PU_discard(epoch, net, p_trainloader, u_trainloader, optimizer,
                                     criterion, device, keep_sample=keep_samples, show_bar=True)
        valid_acc = validate(epoch, net, u_validloader, criterion=criterion, device=device,
                             threshold=0.5, show_bar=True)
        pos_probs = p_probs(net, device, p_validloader)
        unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
        alpha_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
        print(f"[{args.year} seed {args.seed}] epoch {epoch}: train_acc={train_acc} "
              f"valid_acc={valid_acc} alpha_est={alpha_estimate:.3f}")

    # ---- final validation snapshot ----
    pos_probs = p_probs(net, device, p_validloader)
    unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
    y_true = [0] * len(unlabeled_probs) + [1] * len(pos_probs)
    y_scores = (1 - unlabeled_probs[:, 1]).tolist() + pos_probs.tolist()
    auc = roc_auc_score(y_true, y_scores)
    final_mpe, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
    print(f"[{args.year} seed {args.seed}] final val AUC={auc:.3f} val_alpha_est={final_mpe:.3f}")

    model_file = os.path.join(
        log_dir,
        f"TEDn_{args.year}_{args.net_type}_{args.seed}_{args.epochs}_{args.lr}_{args.alpha}_{args.beta}_{timestr}.pt")
    torch.save(net.state_dict(), model_file)
    print(f"saved model -> {model_file}")


if __name__ == "__main__":
    main()
