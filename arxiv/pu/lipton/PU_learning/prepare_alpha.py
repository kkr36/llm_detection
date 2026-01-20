# enter: entrance file, alphas, prior csv (none == make a blank csv, path == append to the existing csv and save to new name? eg have an index 0 that keeps going up each time you pass it in)

entrance_path = "logging_accuracy_alpha_full_sentence"

alphas = [0, .15, .3, .45, .6]

# for each alpha:
    # load in models, given path (both the pt file with pn and the pt file with pu)
    
    # for each model:
        # given path, parse out what experiments you want to rerun/ assemble test sets:
            # if combine, need to fix 2014-2020 (or whatever interval; most recently 2018-2020)
        # for each test set:
            # get: preds (save these in the same folder with year and alpha)
            # bbe (keep upper/lower conf bounds returned by function), 
            # avg pred pos / neg / avg(avg pos, avg neg) / avg(tpr, fpr) aka plugin (bootstrap 90% bounds?)
            # put into a new csv (train_year, train_method, train_alpha, test_alpha, test_year, **test_metrics); save/add to global df



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
from sklearn.metrics import roc_curve, roc_auc_score

font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20
    }
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rc('font', **font)

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
parser.add_argument('--data-dir', type=str, default='/home/ubuntu/data', help='Data directory')
parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer used')
parser.add_argument('--year', type=int, default=None, help='year of arxiv data to take in')
parser.add_argument('--abstract', default=True, action='store_false', help='sentence level analysis')
parser.add_argument('--ft', default=False, action='store_true', help='whether to train on ft or zero shot')
parser.add_argument('--clean', default=False, action='store_true', help='whether to remove chars you cant type on keyboard')
parser.add_argument('--model_path', type=str, default=None, help='path to model')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# to make interpreting things easier
tokenizer = getBertTokenizer('distilbert-base-uncased')
def batch_decode(batch):
    return tokenizer.batch_decode(
        batch,
        skip_special_tokens=True
    )

import numpy as np

def topk_small_large(probs, validdata, k=10):
    order = np.argsort(probs)

    small_idx = order[:k]
    large_idx = order[-k:][::-1]   # largest sorted descending

    small_vals = validdata[small_idx]
    large_vals = validdata[large_idx]

    small_p = probs[small_idx]
    large_p = probs[large_idx]

    return small_vals, small_p, large_vals, large_p


print(args)

net_type = args.net_type
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
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
sentence = args.abstract
ft = args.ft
clean = args.clean

val_alphas = [alpha] if alpha == 0 else [0, alpha]
val_years = [2010, 2012, 2014, 2016, 2018, 2020] if year == 2010 else [year]

# load model

# load data

# inference
    # keep track of whatever stats