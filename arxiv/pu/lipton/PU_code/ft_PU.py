import os
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

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
parser.add_argument('--log-dir', type=str, default='logging_accuracy', help='Dir for logging accuracies')
parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer used')
parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model')

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
log_dir=args.log_dir + "/" + data_type + f"_epochs_{epochs}_lr_{args.lr}_wd_{args.wd}_momentum_{args.momentum}" +"/"
optimizer_str=args.optimizer
alpha_estimate=0.0
show_bar = True
use_alpha = False
data_dir = args.data_dir
estimate_alpha = args.estimate_alpha
model_path = args.model_path

if train_method == "TEDn": 
    use_alpha=True



#################

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestr = time.strftime("%Y%m%d-%H%M%S")

file_name = log_dir + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(train_method, net_type.replace("/", "_"), args.seed, epochs, warm_start_epochs, args.lr, args.wd, args.momentum, alpha, beta)   + "_" + timestr

outfile= open(file_name, 'w')

## Obtain dataset 

if train_method=='PN': 
    u_trainloader, u_validloader, net= get_PN_ft_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size, model_path)
else:
    p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata = \
        get_ft_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size, model_path)

    train_pos_size= len(X)
    train_unlabeled_size= len(Y)
    valid_pos_size= len(p_validdata)
    valid_unlabeled_size= len(u_validdata)

if device.startswith('cuda'):
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if optimizer_str=="SGD":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif optimizer_str=="Adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.wd)
elif optimizer_str=="AdamW": 
    optimizer = AdamW(net.parameters(), lr=args.lr)

## Train in the begining for warm start
if warm_start and train_method=="TEDn": 
    
    outfile.write("Warm_start: \n")

    for epoch in range(warm_start_epochs): 
        train_acc = train(epoch, net, p_trainloader, u_trainloader, \
                optimizer=optimizer, criterion=criterion, device=device, show_bar=show_bar)

        valid_acc = validate(epoch, net, u_validloader, \
                criterion=criterion, device=device, threshold=0.5*beta/(beta + (1-beta)*alpha),show_bar=show_bar)

        if estimate_alpha: 
            pos_probs = p_probs(net, device, p_validloader)
            unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)


            our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)

            dedpul_estimate, dedpul_probs = dedpul(pos_probs, unlabeled_probs,unlabeled_targets)

            EN_estimate= estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
            scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)

            dedpul_accuracy = dedpul_acc(dedpul_probs,unlabeled_targets )*100.0

            alpha_estimate =our_mpe_estimate

        if estimate_alpha:
            outfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc, dedpul_accuracy,\
                 alpha_estimate, dedpul_estimate, EN_estimate, scott_mpe_estimator) )
            outfile.flush()

        else: 
            outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
            outfile.flush()

outfile.write("Algo_training: \n")

if train_method=='PvU': 

    for epoch in range(epochs): 
        if use_alpha: 
            alpha_used = alpha_estimate
        else:
            alpha_used = alpha

        train_acc = train(epoch, net, p_trainloader, u_trainloader, \
                optimizer=optimizer, criterion=criterion, device=device,show_bar=show_bar)

        valid_acc = validate(epoch, net, u_validloader, \
                criterion=criterion, device=device, threshold=0.5*beta/(beta + (1-beta)*alpha_used),show_bar=show_bar)

        if estimate_alpha: 
            pos_probs = p_probs(net, device, p_validloader)
            unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)

            scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)

            our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)

            dedpul_estimate, dedpul_probs = dedpul(pos_probs, unlabeled_probs,unlabeled_targets)

            EN_estimate= estimator_CM_EN(pos_probs, unlabeled_probs[:,0])

            dedpul_accuracy = dedpul_acc(dedpul_probs,unlabeled_targets )*100.0

            alpha_estimate =our_mpe_estimate


            outfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc, dedpul_accuracy,\
                 alpha_estimate, dedpul_estimate, EN_estimate, scott_mpe_estimator) )
            outfile.flush()
        else: 
            outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
            outfile.flush()

elif train_method=='CVIR' or train_method=="TEDn": 

    alpha_used = alpha_estimate

    for epoch in range(epochs):
        
        if use_alpha: 
            alpha_used =  alpha_estimate
        else:
            alpha_used = alpha
        
        keep_samples, neg_reject = rank_inputs(epoch, net, u_trainloader, device,\
             alpha_used, u_size=train_unlabeled_size)
        
        train_acc = train_PU_discard(epoch, net,  p_trainloader, u_trainloader,\
            optimizer, criterion, device, keep_sample=keep_samples,show_bar=show_bar)

        valid_acc, labels, preds = validate(epoch, net, u_validloader, \
            criterion=criterion, device=device, threshold=0.5,show_bar=show_bar)
            
        if estimate_alpha: 
            pos_probs = p_probs(net, device, p_validloader)
            unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)

            our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)

            dedpul_estimate, dedpul_probs = dedpul(pos_probs, unlabeled_probs,unlabeled_targets)

            EN_estimate= estimator_CM_EN(pos_probs, unlabeled_probs[:,0])

            dedpul_accuracy = dedpul_acc(dedpul_probs,unlabeled_targets )*100.0

            alpha_estimate =our_mpe_estimate

            outfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc, dedpul_accuracy,\
                 alpha_estimate, dedpul_estimate, EN_estimate, scott_mpe_estimator) )
            outfile.flush()

        else: 
            outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
            outfile.flush()
            
        model_file = log_dir + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(train_method, net_type.replace("/", "_"), args.seed, epoch, warm_start_epochs, args.lr, args.wd, args.momentum, alpha, beta)   + "_" + timestr
        torch.save(net.state_dict(), f"{model_file}.pt")

elif train_method=='uPU': 

    for epoch in range(epochs):
        
        train_acc = train_PU_unbiased(epoch, net,  p_trainloader, u_trainloader,\
             optimizer, criterion, device, alpha, logistic=(not args.sigmoid_loss), show_bar=show_bar)
            
        valid_acc = validate(epoch, net, u_validloader, \
            criterion=criterion, device=device, threshold=0.5, logistic=(not args.sigmoid_loss), show_bar=show_bar)
    
        outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
        outfile.flush()


elif train_method=='nnPU': 

    for epoch in range(epochs):
        
        train_acc = train_PU_nn_unbiased(epoch, net,  p_trainloader, u_trainloader,\
             optimizer, criterion, device, alpha, logistic=(not args.sigmoid_loss),show_bar=show_bar)
            
        valid_acc = validate(epoch, net, u_validloader, \
            criterion=criterion, device=device, threshold=0.5,logistic=(not args.sigmoid_loss), show_bar=show_bar)
    
        outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
        outfile.flush()

elif train_method=="PN": 

    for epoch in range(epochs):

        train_acc, train_labels, train_preds, train_probs = train_PN(epoch, net, u_trainloader, \
                optimizer=optimizer, criterion=criterion, device=device, show_bar=show_bar)

        valid_acc, labels, preds, probs = validate(epoch, net, u_validloader, \
                criterion=criterion, device=device, threshold=0.5, show_bar=show_bar)

        preds_label_0 = probs[labels == 0]
        preds_label_1 = probs[labels == 1]

        global_min = probs.min()
        global_max = probs.max()
        bins = np.linspace(global_min, global_max, 75)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(preds_label_0, bins=bins, alpha=0.6, color='blue', 
                edgecolor='black', label=f'Label 0 (n={len(preds_label_0)})')
        ax.hist(preds_label_1, bins=bins, alpha=0.6, color='green', 
                edgecolor='black', label=f'Label 1 (n={len(preds_label_1)})')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold=0.5')
        ax.set_xlabel('Prediction Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Prediction Distribution by True Label (Epoch {epoch})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{log_dir}prediction_histograms_epoch{epoch}.pdf', dpi=300, bbox_inches='tight', format='pdf')
        print(f'Saved prediction histograms to prediction_histograms_epoch{epoch}.pdf')

        # get some metrics
        auc = roc_auc_score(labels, probs)
        ce = log_loss(labels, probs)
        tn, fp, fn, tp = confusion_matrix(labels, np.round(probs)).ravel().tolist()
        # import pdb; pdb.set_trace()
        outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc, auc, ce, tn, fp, fn, tp))
        outfile.flush()

        model_file = log_dir + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(train_method, net_type.replace("/", "_"), args.seed, epoch, warm_start_epochs, args.lr, args.wd, args.momentum, alpha, beta)   + "_" + timestr
        torch.save(net.state_dict(), f"{model_file}.pt")


elif train_method=="TiCE" or train_method=="KM": 
    print("here")
    Y_train = u_validdata.data.reshape(len(u_validdata.data), -1)
    X_train = p_validdata.data.reshape(len(p_validdata.data), -1)
    
    
    X = np.concatenate((X,X_train), axis=0)
    Y = np.concatenate((Y,Y_train), axis=0)

    if train_method=="KM":
        print(KM_estimate(X,Y,data_type))
    else: 
        print(TiCE_estimate(X,Y,data_type))

outfile.close()