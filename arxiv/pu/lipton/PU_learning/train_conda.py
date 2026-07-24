"""ConDA training entrypoint (dedicated submodule; train_PU_one_year.py untouched).

Trains one ConDA model whose labeled source = {2020 human, LLM1} and unlabeled
target = {2020 human, LLM2}, given --data-type=llm_type_LLM1|LLM2.

Saves state_dict to  {log_dir}/{data_type}_{epochs}/ConDA_{net}_..._{timestr}.pt
reproducing the directory/filename convention the heatmap eval scripts glob for.
"""
import os
import time
import argparse
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from models.conda import ConDADistilBert
from conda_data import get_conda_loaders
from algorithm_conda import train_conda

parser = argparse.ArgumentParser(description="ConDA Training")
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--wd", default=0.0, type=float)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--data-type", type=str, required=True, help="llm_type_LLM1|LLM2")
parser.add_argument("--train-method", type=str, default="ConDA")
parser.add_argument("--net-type", type=str, default="DistilBert")
parser.add_argument("--optimizer", type=str, default="AdamW")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--alpha", type=float, default=0.5, help="LLM fraction in unlabeled target")
parser.add_argument("--beta", type=float, default=0.6)
parser.add_argument("--year", type=int, default=2020)
parser.add_argument("--abstract", default=True, action="store_false", help="sentence-level when absent")
parser.add_argument("--clean", default=False, action="store_true")
parser.add_argument("--gemini", default=False, action="store_true")
parser.add_argument("--codex", default=False, action="store_true",
                    help="use the codex parquet and treat 'Codex' as a valid LLM column")
parser.add_argument("--flip", default=False, action="store_true", help="human is positive class")
parser.add_argument("--class-weight", default=True, action=argparse.BooleanOptionalAction,
                    help="inverse-frequency weighted CE on the FULL unbalanced source "
                         "(default: on; best AUC + balanced operating point). "
                         "Use --no-class-weight to fall back to subsampling.")
parser.add_argument("--no-balance", dest="balance", default=True, action="store_false",
                    help="only with --no-class-weight: use the raw imbalanced source "
                         "instead of subsampling it to a balanced human/LLM1 set")
parser.add_argument("--log-dir", type=str, default="logging_accuracy_conda")
parser.add_argument("--data-dir", type=str, default="/share/garg/arxiv_kaggle")
# ConDA loss weights
parser.add_argument("--lambda-w", type=float, default=0.5, help="contrastive vs CE weight")
parser.add_argument("--lambda-mmd", type=float, default=1.0, help="MMD domain-alignment weight")
parser.add_argument("--temperature", type=float, default=0.5, help="NT-Xent temperature")
parser.add_argument("--dropout-p", type=float, default=0.2, help="token-dropout augmentation rate")

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
sentence = args.abstract  # store_false: True => sentence-level (matches train_PU_one_year.py)
epochs = args.epochs
data_type = args.data_type

log_dir = args.log_dir + "/" + data_type + "_" + str(epochs) + "/"
os.makedirs(log_dir, exist_ok=True)
timestr = time.strftime("%Y%m%d-%H%M%S")

# --- data ---
# --class-weight uses the full unbalanced source (no subsampling) + weighted CE.
use_balance = args.balance and not args.class_weight
source_loader, target_loader, class_counts = get_conda_loaders(
    args.data_dir, data_type, args.alpha, args.year, sentence,
    args.clean, args.gemini, args.flip, args.seed, batch_size=args.batch_size,
    balance=use_balance, codex=args.codex,
)

class_weight = None
if args.class_weight:
    n0, n1 = class_counts  # (human=class0, LLM1=class1)
    total = n0 + n1
    class_weight = torch.tensor(
        [total / (2.0 * n0), total / (2.0 * n1)], dtype=torch.float, device=device
    )
    print(f"[ConDA] inverse-freq class weights: human={class_weight[0]:.3f}, LLM1={class_weight[1]:.3f}")

# --- model / optimizer ---
net = ConDADistilBert(num_classes=2).to(device)
if device.startswith("cuda"):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

# --- train ---
log_name = log_dir + "ConDA_{}_{}_{}_{}_{}_{}".format(
    args.net_type, args.seed, epochs, args.lr, args.alpha, args.beta) + "_" + timestr
with open(log_name, "w") as outfile:
    outfile.write("epoch, source_acc, ce, ctr, mmd\n")
    for epoch in range(epochs):
        src_acc, ce, ctr, mmd_v = train_conda(
            epoch, net, source_loader, target_loader, optimizer, device,
            lambda_w=args.lambda_w, lambda_mmd=args.lambda_mmd,
            temperature=args.temperature, dropout_p=args.dropout_p,
            class_weight=class_weight,
        )
        outfile.write("{}, {}, {}, {}, {}\n".format(epoch, src_acc, ce, ctr, mmd_v))
        outfile.flush()
        print(f"epoch {epoch}: source_acc={src_acc:.2f} ce={ce:.3f} ctr={ctr:.3f} mmd={mmd_v:.4f}")

# --- save ---
model_file = log_dir + "ConDA_{}_{}_{}_{}_{}_{}".format(
    args.net_type.replace("/", "_"), args.seed, epochs, args.lr, args.alpha, args.beta) + "_" + timestr
torch.save(net.state_dict(), f"{model_file}.pt")
print(f"saved model to {model_file}.pt")
