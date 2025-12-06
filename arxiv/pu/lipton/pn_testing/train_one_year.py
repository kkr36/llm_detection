import os
import argparse
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt

from algorithm import * 
from model_helper import * 
from helper import *
from estimator import *
from baselines import *


np.set_printoptions(suppress=True, precision=1)

parser = argparse.ArgumentParser(description='PU Learning Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--batch-size', type=int, default=200, help='input batch size')
parser.add_argument('--data-type', type=str, help='mnist | cifar')
parser.add_argument('--train-method', type=str, help='training algorithm to use')
parser.add_argument('--net-type', type=str, help='linear | FCN | ResNet')
parser.add_argument('--sigmoid-loss', default=True, action='store_false', help='Sigmoid loss for nnPU training')
parser.add_argument('--estimate-alpha', default=True, action='store_false', help='Estimate alpha')
parser.add_argument('--warm-start', action='store_true', default=False, help='Start domain discrimination training')
parser.add_argument('--warm-start-epochs', type=int, default=0, help='Epochs for domain discrimination training')
parser.add_argument('--epochs', type=int, default=5000, help='Epochs for the specified training algorithm')
parser.add_argument('--seed', type=int, default=42, help='Seed')
parser.add_argument('--alpha', type=float, default=0.5, help='Mixture proportion in unlabeled')
parser.add_argument('--beta', type=float, default=0.5, help='Proportion of labeled in total data ')
parser.add_argument('--log-dir', type=str, default='logging_accuracy_one_year', help='Dir for logging accuracies')
parser.add_argument('--data-dir', type=str, default='/share/garg/arxiv_kaggle', help='Data directory')
parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer used')
parser.add_argument('--year', type=int, default=2010, help='year of arxiv data to take in')
parser.add_argument('--sentence', default=True, action='store_false', help='sentence level analysis')

save_dir_cal = "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/figs"
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print(args)

net_type = args.net_type
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_method = args.train_method
data_type = args.data_type
## Train set for positive and unlabeled
alpha = args.alpha
beta = args.beta
warm_start = args.warm_start
warm_start_epochs = args.warm_start_epochs
batch_size=args.batch_size
epochs=args.epochs
log_dir=args.log_dir + "/" + data_type + "_" + str(epochs) + "/"
optimizer_str=args.optimizer
alpha_estimate=0.0
show_bar = True
use_alpha = False
data_dir = args.data_dir
estimate_alpha = args.estimate_alpha
year = args.year
sentence = args.sentence
# val_alphas = [0.01,.05,.1,.2,.3,.5]
val_alphas = [0, .1, .2, .3, .5][:1]
val_years = [2020, 2023, 2025]

if train_method == "TEDn": 
    use_alpha=True

#################

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestr = time.strftime("%Y%m%d-%H%M%S")

file_name = log_dir + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(train_method, year, net_type, args.seed, epochs, warm_start_epochs, args.lr, args.wd, alpha, beta)   + "_" + timestr

outfile= open(file_name, 'w')

## Obtain dataset 

varied_vals = {}

if train_method=='PN': 
    u_trainloader, u_validloader, net= get_PN_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size, year)
    # import pdb; pdb.set_trace()

else:
    p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata = \
        get_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size, year)
    train_pos_size= len(X)
    train_unlabeled_size= len(Y)
    valid_pos_size= len(p_validdata)
    valid_unlabeled_size= len(u_validdata)

for valyear in val_years:
    varied_vals[valyear] = {}
    for valalpha in tqdm(val_alphas):
        # continue
        p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha = \
            get_dataset_val2(data_dir, data_type,net_type, device, valalpha, beta, batch_size, valyear, sentence)
        varied_vals[valyear][valalpha] = (p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha)


if device.startswith('cuda'):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if optimizer_str=="SGD":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif optimizer_str=="Adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.wd)
elif optimizer_str=="AdamW": 
    # optimizer = AdamW(net.parameters(), lr=args.lr)
    optimizer = AdamW(net.parameters(), lr=args.lr)

outfile.write("Algo_training: \n")

def plot_cal_curves(Y, Y_hat, filepath):
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(Y, Y_hat, strategy='uniform', n_bins=10)

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath, format="pdf")
    plt.clf()


for epoch in tqdm(list(range(epochs))):
    # import pdb; pdb.set_trace()

    train_acc = train_PN(epoch, net, u_trainloader, \
            optimizer=optimizer, criterion=criterion, device=device, show_bar=True)

    valid_acc = validate(epoch, net, u_validloader, \
            criterion=criterion, device=device, threshold=0.5, show_bar=True)

    outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
    outfile.flush()

if estimate_alpha:
    for valalpha in tqdm(val_alphas):
        for valyear in val_years:
            (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[valyear][valalpha]
            pos_probs = p_probs(net, device, p_validloader)
            unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
            naive_mpe_estimate = np.mean(unlabeled_probs[:,0])
            # import pdb; pdb.set_trace()
            our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
            scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
            EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
            outfile.write("{}, {}, {}, {}, {}, {}\n".format(valyear, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, naive_mpe_estimate))
            plot_cal_curves(1-unlabeled_targets, unlabeled_probs[:,0], f"{save_dir_cal}/{year}/calibration_test_{valalpha}_PN.pdf")
