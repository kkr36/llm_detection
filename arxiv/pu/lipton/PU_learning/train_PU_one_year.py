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
parser.add_argument('--data-dir', type=str, default='/share/garg/arxiv_kaggle', help='Data directory')
parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer used')

# new exp args
parser.add_argument('--year', type=int, default=None, help='year of arxiv data to take in')
parser.add_argument('--abstract', default=True, action='store_false', help='sentence level analysis')
parser.add_argument('--ft', default=False, action='store_true', help='whether to train on ft or zero shot')
parser.add_argument('--clean', default=False, action='store_true', help='whether to remove chars you cant type on keyboard')

# llm - llm detection args
parser.add_argument('--gemini', default=False, action='store_true', help='use diverse llms or gemini line')
parser.add_argument('--flip', default=False, action='store_true', help='pos is llm or human')

# line plot args 
parser.add_argument('--combine', default=False, action='store_true', help='use hardcoded years 2014/6/8/20')
parser.add_argument('--add', default=False, action='store_true', help='strictly add positives wrt alpha')
parser.add_argument('--llm', type=str, default=None, help='xy column suffix, e.g. Y or X')


save_dir_cal = "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/figs"
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
gemini = args.gemini
flip = args.flip
combine = args.combine
add = args.add
seed = args.seed
llm = args.llm

# val_alphas = [0.01,.05,.1,.2,.3,.5]
# val_alphas = [0, .1, .25, .5]
# val_alphas = [1, .9, .75, .5, .05]
# val_alphas = [.2, 1]
# val_alphas = [0, .2, .4, .6, .8][-2:-1]
val_alphas = [alpha] if alpha == 0 else [0, alpha]
# val_years = list(range(2010,2026))
val_years = [2010, 2012, 2014, 2016, 2018, 2020] if year == 2010 else [year]
# val_years = [year]

if train_method == "TEDn": 
    use_alpha=True

#################

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestr = time.strftime("%Y%m%d-%H%M%S")

file_name = log_dir + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}".format(train_method, year, net_type, seed, epochs, warm_start_epochs, args.lr, args.wd, alpha, beta, "_ft" if ft else "")   + "_" + timestr

outfile= open(file_name, 'w')

## Obtain dataset 

varied_vals = {}

if train_method=='PN': 
    # import pdb; pdb.set_trace()
    u_trainloader, u_validloader, net= get_PN_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size, year, sentence, ft, clean, gemini, flip, combine, add, seed, llm)
    # import pdb; pdb.set_trace()

else:
    p_trainloader, u_trainloader, p_validloader, u_validloader, p_calloader, u_calloader, net, X, Y, p_validdata, u_validdata, u_traindata = \
        get_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size, year, sentence,ft, clean, gemini, flip, combine, add, seed, llm)
    # import pdb; pdb.set_trace()
    train_pos_size= len(X)
    train_unlabeled_size= len(Y)
    valid_pos_size= len(p_validdata)
    valid_unlabeled_size= len(u_validdata)

# import pdb; pdb.set_trace()
if data_type=="ArXiv_BERT":
    for valyear in val_years:
        varied_vals[valyear] = {}
        for valalpha in val_alphas:
        # valalpha = .12 * ((valyear - 2010) // 2)
        # valalpha = 0
        # for valalpha in tqdm(val_alphas):
            # continue
            p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha = \
                get_dataset_val2(data_dir, data_type,net_type, device, valalpha, beta, batch_size, valyear, sentence, ft, clean, gemini, flip, combine, add, seed)
            varied_vals[valyear][valalpha] = (p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha)
elif data_type=="paramveer":
    varied_vals['ft'] = {}
    varied_vals['ai'] = {}

    for alpha in val_alphas:
        varied_vals['ft'][alpha] = get_dataset_val2(data_dir, data_type,net_type, device, alpha, None, batch_size, None, None, ft=True, clean=clean, gemini=gemini, flip=flip, combine=combine, add=add, seed=seed)
        varied_vals['ai'][alpha] = get_dataset_val2(data_dir, data_type,net_type, device, alpha, None, batch_size, None, None, ft=False, clean=clean, gemini=gemini, flip=flip, combine=combine, add=add, seed=seed)
elif "llm_type_" in data_type:
    llm_list = ["Gemini 3 Preview", "Qwen", "GPT OSS 120b", "Llama 3.3 70b Instruct"] if not gemini else ["Gemini 2.0 Flash-Lite", "Gemini 2.0 Flash", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Preview"]
    for llm in tqdm(llm_list):
        varied_vals[llm] = {}
        for valalpha in val_alphas:
            p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha = \
                get_dataset_val2(data_dir, f"llm_type_{llm.replace(' ', '_')}", net_type, device, valalpha, beta, batch_size, year, sentence, ft, clean, gemini, flip, combine, add, seed)
            varied_vals[llm][valalpha] = (p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha)
elif "Arxiv_year" in data_type:
    for valalpha in val_alphas:
        p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha = \
            get_dataset_val2(data_dir, data_type,net_type, device, valalpha, beta, batch_size, None, sentence, ft, clean, gemini, flip, combine, add, seed)
        varied_vals[valalpha] = (p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha)
elif data_type == "xy":
    for valalpha in val_alphas:
        p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha = \
            get_dataset_val2(data_dir, data_type, net_type, device, valalpha, beta, batch_size, year, sentence, ft, clean, gemini, flip, combine, add, seed, llm)
        varied_vals[valalpha] = (p_validloader_alpha, u_validloader_alpha, p_validdata_alpha, u_validdata_alpha)

# import pdb; pdb.set_trace()

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

            # dedpul_estimate, dedpul_probs = dedpul(pos_probs, unlabeled_probs,unlabeled_targets)

            EN_estimate= estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
            scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)

            # dedpul_accuracy = dedpul_acc(dedpul_probs,unlabeled_targets )*100.0

            alpha_estimate =our_mpe_estimate

        if estimate_alpha:
            # outfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc, dedpul_accuracy,\
            #      alpha_estimate, dedpul_estimate, EN_estimate, scott_mpe_estimator) )
            # outfile.flush()
            outfile.write("{}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc,\
                 alpha_estimate, EN_estimate, scott_mpe_estimator) )
            outfile.flush()

        else: 
            outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
            outfile.flush()

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

            # dedpul_estimate, dedpul_probs = dedpul(pos_probs, unlabeled_probs,unlabeled_targets)

            EN_estimate= estimator_CM_EN(pos_probs, unlabeled_probs[:,0])

            # dedpul_accuracy = dedpul_acc(dedpul_probs,unlabeled_targets )*100.0

            alpha_estimate =our_mpe_estimate


            outfile.write("{}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc,\
                 alpha_estimate, EN_estimate, scott_mpe_estimator) )
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
        # import pdb; pdb.set_trace()
        valid_acc = validate(epoch, net, u_validloader, \
            criterion=criterion, device=device, threshold=0.5,show_bar=show_bar)
        
        # if not os.path.exists(f"{save_dir_cal}/{year}"):
        #     os.makedirs(f"{save_dir_cal}/{year}")

        if estimate_alpha: 
            # continue
            pos_probs = p_probs(net, device, p_validloader)
            unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
            # plot_cal_curves(unlabeled_targets, unlabeled_probs[:,1], f"{save_dir_cal}/{year}/calibration_test.pdf")

            our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)

            # dedpul_estimate, dedpul_probs = dedpul(pos_probs, unlabeled_probs,unlabeled_targets)
            scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)

            EN_estimate= estimator_CM_EN(pos_probs, unlabeled_probs[:,0])

            # dedpul_accuracy = dedpul_acc(dedpul_probs,unlabeled_targets )*100.0

            alpha_estimate =our_mpe_estimate

            if data_type=="ArXiv_BERT":
                ## Cal set of uncontaminated data
                cal_acc = validate(epoch, net, u_calloader, \
                    criterion=criterion, device=device, threshold=0.5,show_bar=show_bar)
                cal_pos_probs = p_probs(net, device, p_calloader)
                cal_unlabeled_probs, cal_unlabeled_targets = u_probs(net, device, u_calloader)
                cal_mpe_estimate, _, _ = BBE_estimator(cal_pos_probs, cal_unlabeled_probs, cal_unlabeled_targets)
                # import pdb; pdb.set_trace()
                # outfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc, cal_acc, cal_mpe_estimate,\
                #      alpha_estimate, EN_estimate, scott_mpe_estimator) )
                outfile.write("{}, {}, {}, {}, {}, {}\n".format(epoch, train_acc, valid_acc, cal_acc, cal_mpe_estimate,\
                    alpha_estimate) )
            else:
                outfile.write("{}, {}, {}, {}\n".format(epoch, train_acc, valid_acc,\
                    alpha_estimate) )
            outfile.flush()

        else: 
            outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
            outfile.flush()
    if estimate_alpha and data_type=="ArXiv_BERT":
        for valyear in val_years:
            for valalpha in varied_vals[valyear]:
                (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[valyear][valalpha]
                pos_probs = p_probs(net, device, p_validloader) # for me right now, preds over garbage watermark
                pos_prob = np.mean(pos_probs)
                unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader) # for me right now, preds over actual abstracts
                preds = np.argmax(unlabeled_probs, axis=1)
                neg_probs = 1-unlabeled_probs[:,1]
                neg_prob = np.mean(neg_probs)

                y_true = [0 for _ in range(len(unlabeled_probs))] + [1 for _ in range(len(pos_probs))]
                y_scores = (1-unlabeled_probs[:,1]).tolist() + pos_probs.tolist()
                auc = roc_auc_score(y_true, y_scores)
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)

                # n = 15
                # small_pos_val, small_pos_prob, large_pos_val, large_pos_prob = topk_small_large(pos_probs, p_validdata.data, n)
                # small_neg_val, small_neg_prob, large_neg_val, large_neg_prob = topk_small_large(neg_probs, u_validdata.data, n)
                # small_pos_text, large_pos_text = batch_decode(small_pos_val[:,:,0]), batch_decode(large_pos_val[:,:,0])
                # small_neg_text, large_neg_text = batch_decode(small_neg_val[:,:,0]), batch_decode(large_neg_val[:,:,0])
                # texts = small_pos_text + large_pos_text + small_neg_text + large_neg_text
                # probs = small_pos_prob.tolist() + large_pos_prob.tolist() + small_neg_prob.tolist() + large_neg_prob.tolist()
                # labels = [1 for _ in range(n*2)] + [0 for _ in range(n*2)]

                # df = pd.DataFrame({
                #     "text": texts,
                #     "prob": probs,
                #     "label": labels
                # })

                # import pdb; pdb.set_trace()

                # df.to_csv(f"/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/logging_accuracy_pdb/test_PU.csv")
                # import pdb; pdb.set_trace()

                pos_lens = [sum(x[:,1]) for x in p_validdata.data]
                neg_lens = [sum(x[:,1]) for x in u_validdata.data]

                plt.scatter(pos_lens, pos_probs, label="Positives")
                plt.scatter(neg_lens, neg_probs, label="Negatives")
                plt.xlabel("Length")
                plt.ylabel("P(LLM)")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}len_prob_{train_method}_{'sentence' if sentence else 'abstract'}_{valyear}_{valalpha}.pdf", format="pdf")
                plt.clf()

                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}auc_{train_method}_{'sentence' if sentence else 'abstract'}_{valyear}_{valalpha}.pdf", format='pdf')
                plt.clf()

                bins = np.linspace(
                    min(pos_probs.min(), neg_probs.min()),
                    max(pos_probs.max(), neg_probs.max()),
                    50
                )

                plt.figure()
                plt.hist(pos_probs, bins=bins, alpha=0.3, density=True, label="Positive")
                plt.hist(neg_probs, bins=bins, alpha=0.3, density=True, label="Negative")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}hists_{train_method}_{'sentence' if sentence else 'abstract'}_{valyear}_{valalpha}.pdf", format='pdf')
                plt.clf()

                actual_mpe = 1 - np.mean(unlabeled_targets)
                neg_acc = np.mean(preds == unlabeled_targets)
                pos_acc = np.mean(np.round(pos_probs) == 1) 
                our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
                scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
                EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
                # import pdb; pdb.set_trace()

                outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(valyear, actual_mpe, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, neg_acc, neg_prob, pos_acc, pos_prob, auc))
                # plot_cal_curves(1-unlabeled_targets, unlabeled_probs[:,0], f"{save_dir_cal}/{year}/calibration_test_{valalpha}.pdf")
    elif estimate_alpha and "llm_type_" in data_type:
        llm = data_type.split("llm_type_")[-1]
        for llm_ood in varied_vals:
            for valalpha in val_alphas:
                (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[llm_ood][valalpha]
                pos_probs = p_probs(net, device, p_validloader)
                pos_prob = np.mean(pos_probs)
                unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
                neg_probs = 1-unlabeled_probs[:,1]
                neg_prob = np.mean(neg_probs)
                naive_mpe_estimate = np.mean(unlabeled_probs[:,0])
                preds = np.argmax(unlabeled_probs, axis=1) # TODO change to average prob on the class, or cross entropy

                # import pdb; pdb.set_trace()

                y_true = [0 for _ in range(len(unlabeled_probs))] + [1 for _ in range(len(pos_probs))]
                y_scores = (1-unlabeled_probs[:,1]).tolist() + pos_probs.tolist()
                auc = roc_auc_score(y_true, y_scores)
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)

                bins = np.linspace(
                    min(pos_probs.min(), neg_probs.min()),
                    max(pos_probs.max(), neg_probs.max()),
                    50
                )

                plt.figure()
                plt.hist(pos_probs, bins=bins, alpha=0.3, density=True, label="Positive")
                plt.hist(neg_probs, bins=bins, alpha=0.3, density=True, label="Negative")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}hists_{train_method}_{llm_ood}_{valalpha}.pdf", format='pdf')
                plt.clf()

                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}auc_{train_method}_{llm_ood}_{valalpha}.pdf", format='pdf')
                plt.clf()

                neg_acc = np.mean(preds == unlabeled_targets)
                pos_acc = np.mean(np.round(pos_probs) == 1)
                actual_mpe = 1 - np.mean(unlabeled_targets)
                our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
                scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
                EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
                outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(llm_ood, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, naive_mpe_estimate, neg_acc, neg_prob, pos_acc, pos_prob, auc))  
    elif estimate_alpha and data_type=="paramveer":
        for key in varied_vals:
            for valalpha in varied_vals[key]:
                (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[key][valalpha]
                # import pdb; pdb.set_trace()
                pos_probs = p_probs(net, device, p_validloader)
                unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
                # import pdb; pdb.set_trace()
                our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
                scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
                EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
                outfile.write("{}, {}, {}, {}, {}\n".format(key, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate))
    elif estimate_alpha and "Arxiv_year" in data_type:
        for valalpha in varied_vals:
            (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[valalpha]
            pos_probs = p_probs(net, device, p_validloader) # for me right now, preds over garbage watermark
            pos_prob = np.mean(pos_probs)
            unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader) # for me right now, preds over actual abstracts
            preds = np.argmax(unlabeled_probs, axis=1)
            neg_probs = 1-unlabeled_probs[:,1]
            neg_prob = np.mean(neg_probs)

            y_true = [0 for _ in range(len(unlabeled_probs))] + [1 for _ in range(len(pos_probs))]
            y_scores = (1-unlabeled_probs[:,1]).tolist() + pos_probs.tolist()
            auc = roc_auc_score(y_true, y_scores)
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)

            pos_lens = [sum(x[:,1]) for x in p_validdata.data]
            neg_lens = [sum(x[:,1]) for x in u_validdata.data]

            plt.scatter(pos_lens, pos_probs, label="Positives")
            plt.scatter(neg_lens, neg_probs, label="Negatives")
            plt.xlabel("Length")
            plt.ylabel("P(LLM)")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{log_dir}len_prob_{train_method}_{'sentence' if sentence else 'abstract'}_{valalpha}.pdf", format="pdf")
            plt.clf()

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{log_dir}auc_{train_method}_{'sentence' if sentence else 'abstract'}_{valalpha}.pdf", format='pdf')
            plt.clf()

            bins = np.linspace(
                min(pos_probs.min(), neg_probs.min()),
                max(pos_probs.max(), neg_probs.max()),
                50
            )

            plt.figure()
            plt.hist(pos_probs, bins=bins, alpha=0.3, density=True, label="Positive")
            plt.hist(neg_probs, bins=bins, alpha=0.3, density=True, label="Negative")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{log_dir}hists_{train_method}_{'sentence' if sentence else 'abstract'}_{valalpha}.pdf", format='pdf')
            plt.clf()

            actual_mpe = 1 - np.mean(unlabeled_targets)
            neg_acc = np.mean(preds == unlabeled_targets)
            pos_acc = np.mean(np.round(pos_probs) == 1)
            our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
            scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
            EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
            # import pdb; pdb.set_trace()

            outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(actual_mpe, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, neg_acc, neg_prob, pos_acc, pos_prob, auc))
            # plot_cal_curves(1-unlabeled_targets, unlabeled_probs[:,0], f"{save_dir_cal}/{year}/calibration_test_{valalpha}.pdf")
    elif estimate_alpha and data_type == "xy":
        for valalpha in varied_vals:
            (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[valalpha]
            pos_probs = p_probs(net, device, p_validloader)
            pos_prob = np.mean(pos_probs)
            unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
            preds = np.argmax(unlabeled_probs, axis=1)
            neg_probs = 1-unlabeled_probs[:,1]
            neg_prob = np.mean(neg_probs)

            y_true = [0 for _ in range(len(unlabeled_probs))] + [1 for _ in range(len(pos_probs))]
            y_scores = (1-unlabeled_probs[:,1]).tolist() + pos_probs.tolist()
            auc = roc_auc_score(y_true, y_scores)
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)

            bins = np.linspace(
                min(pos_probs.min(), neg_probs.min()),
                max(pos_probs.max(), neg_probs.max()),
                50
            )

            plt.figure()
            plt.hist(pos_probs, bins=bins, alpha=0.3, density=True, label="Positive")
            plt.hist(neg_probs, bins=bins, alpha=0.3, density=True, label="Negative")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{log_dir}hists_{train_method}_{'sentence' if sentence else 'abstract'}_{valalpha}.pdf", format='pdf')
            plt.clf()

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{log_dir}auc_{train_method}_{'sentence' if sentence else 'abstract'}_{valalpha}.pdf", format='pdf')
            plt.clf()

            actual_mpe = 1 - np.mean(unlabeled_targets)
            neg_acc = np.mean(preds == unlabeled_targets)
            pos_acc = np.mean(np.round(pos_probs) == 1)
            our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
            scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
            EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])

            outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(actual_mpe, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, neg_acc, neg_prob, pos_acc, pos_prob, auc))

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

    for epoch in tqdm(list(range(epochs))):
        # import pdb; pdb.set_trace()

        train_acc = train_PN(epoch, net, u_trainloader, \
                optimizer=optimizer, criterion=criterion, device=device, show_bar=True)

        valid_acc = validate(epoch, net, u_validloader, \
                criterion=criterion, device=device, threshold=0.5, show_bar=True)

        outfile.write("{}, {}, {}\n".format(epoch, train_acc, valid_acc))
        outfile.flush()

    if estimate_alpha and data_type=="ArXiv_BERT":
        for valyear in val_years:
            for valalpha in varied_vals[valyear]:
                # contamination_pct = 
                (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[valyear][valalpha]
                pos_probs = p_probs(net, device, p_validloader)
                pos_prob = np.mean(pos_probs)
                unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
                neg_probs = 1-unlabeled_probs[:,1]
                neg_prob = np.mean(neg_probs)
                naive_mpe_estimate = np.mean(unlabeled_probs[:,0])
                preds = np.argmax(unlabeled_probs, axis=1) # TODO change to average prob on the class, or cross entropy

                y_true = [0 for _ in range(len(unlabeled_probs))] + [1 for _ in range(len(pos_probs))]
                y_scores = (1-unlabeled_probs[:,1]).tolist() + pos_probs.tolist()
                auc = roc_auc_score(y_true, y_scores)
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                # import pdb; pdb.set_trace()

                bins = np.linspace(
                    min(pos_probs.min(), neg_probs.min()),
                    max(pos_probs.max(), neg_probs.max()),
                    50
                )

                plt.figure()
                plt.hist(pos_probs, bins=bins, alpha=0.3, density=True, label="Positive")
                plt.hist(neg_probs, bins=bins, alpha=0.3, density=True, label="Negative")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}hists_{train_method}_{'sentence' if sentence else 'abstract'}_{valyear}_{valalpha}.pdf", format='pdf')
                plt.clf()

                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}auc_{train_method}_{'sentence' if sentence else 'abstract'}_{valyear}_{valalpha}.pdf", format='pdf')
                plt.clf()

                neg_acc = np.mean(preds == unlabeled_targets)
                pos_acc = np.mean(np.round(pos_probs) == 1)
                actual_mpe = 1 - np.mean(unlabeled_targets)
                # import pdb; pdb.set_trace()
                our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
                scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
                EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
                
                # n = 15
                # small_pos_val, small_pos_prob, large_pos_val, large_pos_prob = topk_small_large(pos_probs, p_validdata.data, n)
                # small_neg_val, small_neg_prob, large_neg_val, large_neg_prob = topk_small_large(neg_probs, u_validdata.data, n)
                # small_pos_text, large_pos_text = batch_decode(small_pos_val[:,:,0]), batch_decode(large_pos_val[:,:,0])
                # small_neg_text, large_neg_text = batch_decode(small_neg_val[:,:,0]), batch_decode(large_neg_val[:,:,0])
                # texts = small_pos_text + large_pos_text + small_neg_text + large_neg_text
                # probs = small_pos_prob.tolist() + large_pos_prob.tolist() + small_neg_prob.tolist() + large_neg_prob.tolist()
                # labels = [1 for _ in range(n*2)] + [0 for _ in range(n*2)]

                # df = pd.DataFrame({
                #     "text": texts,
                #     "prob": probs,
                #     "label": labels
                # })

                # df.to_csv(f"/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning/logging_accuracy_pdb/test.csv")
                # import pdb; pdb.set_trace()
                # quit()

                outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(valyear, actual_mpe, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, naive_mpe_estimate, neg_acc, neg_prob, pos_acc, pos_prob, auc))
                # outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format())
                # outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(valyear, actual_mpe, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, neg_acc, neg_prob, pos_acc, pos_prob))

                # plot_cal_curves(1-unlabeled_targets, unlabeled_probs[:,0], f"{save_dir_cal}/{year}/calibration_test_{valalpha}_PN.pdf")
    elif estimate_alpha and "llm_type_" in data_type:
        llm = data_type.split("llm_type_")[-1]
        for llm_ood in varied_vals:
            for valalpha in val_alphas:
                (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[llm_ood][valalpha]
                pos_probs = p_probs(net, device, p_validloader)
                pos_prob = np.mean(pos_probs)
                unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
                neg_probs = 1-unlabeled_probs[:,1]
                neg_prob = np.mean(neg_probs)
                naive_mpe_estimate = np.mean(unlabeled_probs[:,0])
                preds = np.argmax(unlabeled_probs, axis=1) # TODO change to average prob on the class, or cross entropy

                y_true = [0 for _ in range(len(unlabeled_probs))] + [1 for _ in range(len(pos_probs))]
                y_scores = (1-unlabeled_probs[:,1]).tolist() + pos_probs.tolist()
                auc = roc_auc_score(y_true, y_scores)
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)

                bins = np.linspace(
                    min(pos_probs.min(), neg_probs.min()),
                    max(pos_probs.max(), neg_probs.max()),
                    50
                )

                plt.figure()
                plt.hist(pos_probs, bins=bins, alpha=0.3, density=True, label="Positive")
                plt.hist(neg_probs, bins=bins, alpha=0.3, density=True, label="Negative")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}hists_{train_method}_{llm_ood}.pdf", format='pdf')
                plt.clf()

                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.tight_layout()

                plt.savefig(f"{log_dir}auc_{train_method}_{llm_ood}.pdf", format='pdf')
                plt.clf()

                neg_acc = np.mean(preds == unlabeled_targets)
                pos_acc = np.mean(np.round(pos_probs) == 1)
                actual_mpe = 1 - np.mean(unlabeled_targets)
                our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
                scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
                EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
                outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(llm_ood, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, naive_mpe_estimate, neg_acc, neg_prob, pos_acc, pos_prob, auc))
                # import pdb; pdb.set_trace()

    elif estimate_alpha and data_type=="paramveer":
        for key in varied_vals:
            for valalpha in varied_vals[key]:
                (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[key][valalpha]
                pos_probs = p_probs(net, device, p_validloader)
                unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
                # import pdb; pdb.set_trace()
                naive_mpe_estimate = np.mean(unlabeled_probs[:,0])
                our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
                scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
                EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
                outfile.write("{}, {}, {}, {}, {}, {}\n".format(key, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, naive_mpe_estimate))
    elif estimate_alpha and data_type == "xy":
        for valalpha in varied_vals:
            (p_validloader, u_validloader, p_validdata, u_validdata) = varied_vals[valalpha]
            pos_probs = p_probs(net, device, p_validloader)
            pos_prob = np.mean(pos_probs)
            unlabeled_probs, unlabeled_targets = u_probs(net, device, u_validloader)
            neg_probs = 1-unlabeled_probs[:,1]
            neg_prob = np.mean(neg_probs)
            naive_mpe_estimate = np.mean(unlabeled_probs[:,0])
            preds = np.argmax(unlabeled_probs, axis=1)

            y_true = [0 for _ in range(len(unlabeled_probs))] + [1 for _ in range(len(pos_probs))]
            y_scores = (1-unlabeled_probs[:,1]).tolist() + pos_probs.tolist()
            auc = roc_auc_score(y_true, y_scores)
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)

            bins = np.linspace(
                min(pos_probs.min(), neg_probs.min()),
                max(pos_probs.max(), neg_probs.max()),
                50
            )

            plt.figure()
            plt.hist(pos_probs, bins=bins, alpha=0.3, density=True, label="Positive")
            plt.hist(neg_probs, bins=bins, alpha=0.3, density=True, label="Negative")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{log_dir}hists_{train_method}_{'sentence' if sentence else 'abstract'}_{valalpha}.pdf", format='pdf')
            plt.clf()

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{log_dir}auc_{train_method}_{'sentence' if sentence else 'abstract'}_{valalpha}.pdf", format='pdf')
            plt.clf()

            neg_acc = np.mean(preds == unlabeled_targets)
            pos_acc = np.mean(np.round(pos_probs) == 1)
            actual_mpe = 1 - np.mean(unlabeled_targets)
            our_mpe_estimate, _, _ = BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
            scott_mpe_estimator = scott_estimator(pos_probs, unlabeled_probs)
            EN_estimate = estimator_CM_EN(pos_probs, unlabeled_probs[:,0])
            outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(actual_mpe, valalpha, our_mpe_estimate, scott_mpe_estimator, EN_estimate, naive_mpe_estimate, neg_acc, neg_prob, pos_acc, pos_prob, auc))

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

# TODO save net here
model_file = log_dir + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(train_method, net_type.replace("/", "_"), args.seed, epoch, warm_start_epochs, args.lr, args.wd, args.momentum, alpha, beta)   + "_" + timestr
torch.save(net.state_dict(), f"{model_file}.pt")
print(f"saved model to {model_file}.pt")