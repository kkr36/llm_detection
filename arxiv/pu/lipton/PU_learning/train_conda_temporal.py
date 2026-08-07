"""ConDA training entrypoint for the TEMPORAL setting (year-to-year human shift).

Sibling of train_conda.py; train_conda.py and train_PU_one_year.py are untouched.
Reuses models/conda.py (ConDADistilBert) and algorithm_conda.py (train_conda)
unchanged. The only difference from train_conda.py is the data source: here the
labeled source = {AI test-year, 2010 human} and unlabeled target = the test-year
mix, built by conda_data_temporal.get_conda_temporal_loaders (which wraps the PNU
temporal reader read_arxiv_split2_PNU). No --flip: class 0 = AI, matching how
prepare_temporal.py evaluates the PNU temporal models.

Saves state_dict to  {log_dir}/ArXiv_BERT_{epochs}/ConDA_{net}_..._{timestr}.pt
paralleling the PNU temporal tree logging_accuracy_temporal_alpha_full_sentence/.
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
from conda_data_temporal import get_conda_temporal_loaders
from algorithm_conda import train_conda

parser = argparse.ArgumentParser(description="ConDA Temporal Training")
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--wd", default=0.0, type=float)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--train-method", type=str, default="ConDA")
parser.add_argument("--net-type", type=str, default="DistilBert")
parser.add_argument("--optimizer", type=str, default="AdamW")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--alpha", type=float, default=0.5, help="AI fraction in unlabeled target")
parser.add_argument("--beta", type=float, default=0.6)
parser.add_argument("--year", type=int, default=2020, help="test (target) year")
parser.add_argument("--abstract", default=True, action="store_false", help="sentence-level when absent")
parser.add_argument("--clean", default=False, action="store_true")
parser.add_argument("--class-weight", default=True, action=argparse.BooleanOptionalAction,
                    help="inverse-frequency weighted CE on the FULL unbalanced source "
                         "(default: on). Use --no-class-weight to fall back to subsampling.")
parser.add_argument("--no-balance", dest="balance", default=True, action="store_false",
                    help="only with --no-class-weight: use the raw imbalanced source "
                         "instead of subsampling it to a balanced AI/human set")
parser.add_argument("--log-dir", type=str, default="logging_accuracy_temporal_alpha_full_sentence_conda")
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
data_type = "ArXiv_BERT"

log_dir = args.log_dir + "/" + data_type + "_" + str(epochs) + "/"
os.makedirs(log_dir, exist_ok=True)
timestr = time.strftime("%Y%m%d-%H%M%S")

# --- data ---
# --class-weight uses the full unbalanced source (no subsampling) + weighted CE.
use_balance = args.balance and not args.class_weight
source_loader, target_loader, class_counts = get_conda_temporal_loaders(
    args.data_dir, args.alpha, args.year, sentence, args.clean, args.seed,
    batch_size=args.batch_size, balance=use_balance,
)

class_weight = None
if args.class_weight:
    n0, n1 = class_counts  # (AI=class0, human2010=class1)
    total = n0 + n1
    class_weight = torch.tensor(
        [total / (2.0 * n0), total / (2.0 * n1)], dtype=torch.float, device=device
    )
    print(f"[ConDA temporal] inverse-freq class weights: AI={class_weight[0]:.3f}, human2010={class_weight[1]:.3f}")

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
