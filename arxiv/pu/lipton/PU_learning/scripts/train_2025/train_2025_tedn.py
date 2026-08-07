"""
Train a single TEDn model on the 2025 back-half arXiv data, in the same style as
train_PU_one_year.py (data-type ArXiv_BERT, sentence-level, DistilBert, AdamW,
lr=1e-5, beta=.6, epochs=3, --clean), but with:

  - labeled positives = 2025 LLM mirrors
  - unlabeled         = 2025 human_abstract column + alpha-fraction injected mirrors

This is a NEW standalone script: it imports the existing training building blocks
(algorithm / model_helper / helper / estimator / baselines) but does not modify
any existing file.  It reproduces the TEDn ("CVIR"/TEDn) training loop and saves
the trained net to --log-dir.

Run one model:
  python scripts/train_2025/train_2025_tedn.py --seed 0
Or use the slurm array (5 seeds) in this directory.
"""

import os
import sys
import time
import random
import argparse

# make the PU_learning root importable no matter where we launch from
PU_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PU_ROOT not in sys.path:
    sys.path.insert(0, PU_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score

from algorithm import *          # train_PU_discard, rank_inputs, validate, train
from model_helper import *       # get_model, initialize_bert_transform (via helper)
from helper import *             # IMDbBERTData, get_PUDataSplits1, clean_text, shuffle, initialize_bert_transform
from estimator import *          # p_probs, u_probs, BBE_estimator
from baselines import *          # scott_estimator, estimator_CM_EN

from scripts.train_2025.read_2025 import read_arxiv_2025_backhalf

np.set_printoptions(suppress=True, precision=1)


def build_loaders(data_path, alpha, seed, sentence, clean):
    """Build P/U train + val loaders from the 2025 back-half parquet."""
    transform = initialize_bert_transform("distilbert-base-uncased")

    train_texts, train_labels = read_arxiv_2025_backhalf(data_path, alpha, "train", sentence, seed)
    val_texts, val_labels = read_arxiv_2025_backhalf(data_path, alpha, "val", sentence, seed)

    if clean:
        orig = sum(len(x) for x in train_texts)
        train_texts = clean_text(train_texts)
        print(f"cleaned train text of {orig - sum(len(x) for x in train_texts)} funny chars")
        val_texts = clean_text(val_texts)

    train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
    val_dataset = IMDbBERTData(val_texts, val_labels, transform=transform)

    p_traindata, u_traindata = get_PUDataSplits1(train_dataset, data_type="ArXiv2025_backhalf")
    p_validdata, u_validdata = get_PUDataSplits1(val_dataset, data_type="ArXiv2025_backhalf")

    p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=8, shuffle=shuffle)
    u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=8, shuffle=shuffle)
    p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=128, shuffle=shuffle)
    u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, shuffle=shuffle)

    return (p_trainloader, u_trainloader, p_validloader, u_validloader,
            p_traindata, u_traindata, p_validdata, u_validdata)


def main():
    parser = argparse.ArgumentParser(description="Train TEDn on 2025 back-half arXiv (mirrors=P, humans+alpha mirrors=U)")
    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--momentum", default=0.0, type=float)
    parser.add_argument("--alpha", default=0.0, type=float,
                        help="injected-mirror fraction of the unlabeled pool; 0 = unlabeled is purely 2025 'human' text")
    parser.add_argument("--beta", default=0.6, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    # same semantics as train_PU_one_year.py: sentence-level is the default (--abstract
    # flips to abstract-level); --clean is opt-in and passed by run_arxiv.py / the sbatch.
    parser.add_argument("--abstract", dest="sentence", default=True, action="store_false",
                        help="pass for abstract-level; default (omitted) is sentence-level, matching run_arxiv.py")
    parser.add_argument("--clean", default=False, action="store_true",
                        help="strip non-typable chars (run_arxiv.py passes this)")
    parser.add_argument("--data-path", type=str,
                        default="/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_2025_ai_cs._10000_backhalf.parquet")
    parser.add_argument("--log-dir", type=str, default="/share/garg/arxiv_kaggle/2025_models")
    parser.add_argument("--net-type", type=str, default="DistilBert")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_type = args.net_type
    alpha = args.alpha
    beta = args.beta
    epochs = args.epochs
    data_type = "ArXiv2025_backhalf"

    log_dir = args.log_dir + "/" + data_type + "_" + str(epochs) + "/"
    os.makedirs(log_dir, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")

    file_name = log_dir + "TEDn_2025_{}_{}_{}_{}_{}_{}_{}".format(
        net_type, args.seed, epochs, args.lr, args.wd, alpha, beta) + "_" + timestr
    outfile = open(file_name, "w")

    (p_trainloader, u_trainloader, p_validloader, u_validloader,
     p_traindata, u_traindata, p_validdata, u_validdata) = build_loaders(
        args.data_path, alpha, args.seed, args.sentence, args.clean)

    train_unlabeled_size = len(u_traindata.targets)

    net = get_model(net_type).to(device)
    if device.startswith("cuda"):
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(net.parameters(), lr=args.lr)

    # ---- TEDn training loop (mirrors the CVIR/TEDn branch of train_PU_one_year.py) ----
    outfile.write("Algo_training: \n")
    alpha_estimate = 0.0
    use_alpha = True  # TEDn estimates alpha from the data
    for epoch in range(epochs):
        alpha_used = alpha_estimate if use_alpha else alpha

        keep_samples, neg_reject = rank_inputs(
            epoch, net, u_trainloader, device, alpha_used, u_size=train_unlabeled_size)

        train_acc = train_PU_discard(
            epoch, net, p_trainloader, u_trainloader, optimizer, criterion, device,
            keep_sample=keep_samples, show_bar=True)

        valid_acc = validate(
            epoch, net, u_validloader, criterion=criterion, device=device,
            threshold=0.5, show_bar=True)

        pos_probs = p_probs(net, device, p_validloader)
        unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
        our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
        alpha_estimate = our_mpe_estimate

        outfile.write("{}, {}, {}, {}\n".format(epoch, train_acc, valid_acc, alpha_estimate))
        outfile.flush()
        print(f"[seed {args.seed}] epoch {epoch}: train_acc={train_acc} valid_acc={valid_acc} "
              f"alpha_est={alpha_estimate:.3f}")

    # ---- final validation metrics (P = mirrors vs U-negatives = humans) ----
    pos_probs = p_probs(net, device, p_validloader)
    unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
    neg_probs = 1 - unlabeled_probs[:, 1]
    y_true = [0] * len(unlabeled_probs) + [1] * len(pos_probs)
    y_scores = (1 - unlabeled_probs[:, 1]).tolist() + pos_probs.tolist()
    auc = roc_auc_score(y_true, y_scores)
    our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
    scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
    EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:, 0])
    outfile.write("final, {}, {}, {}, {}, {}, {}\n".format(
        alpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate,
        float(np.mean(pos_probs)), auc))
    outfile.flush()
    outfile.close()
    print(f"[seed {args.seed}] final AUC={auc:.3f} alpha_est={our_mpe_estimate:.3f}")

    # ---- save model ----
    model_file = log_dir + "TEDn_2025_{}_{}_{}_{}_{}_{}_{}".format(
        net_type.replace("/", "_"), args.seed, epochs, args.lr, args.wd, alpha, beta) + "_" + timestr
    torch.save(net.state_dict(), f"{model_file}.pt")
    print(f"saved model to {model_file}.pt")


if __name__ == "__main__":
    main()
