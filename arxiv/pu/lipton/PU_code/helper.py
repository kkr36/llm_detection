import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

from data_helper import * 
from model_helper import * 


class PosData(torch.utils.data.Dataset): 
    def __init__(self, transform=None, target_transform=None, data=None, \
            index=None, data_type=None):
        self.transform = transform
        self.target_transform = target_transform

        self.data=data
        self.targets = np.zeros(data.shape[0], dtype= np.int_)
        self.data_type = data_type
        self.index = index

    def __len__(self): 
        return len(self.targets)

    def __getitem__(self, idx):
        index, img, target = self.index[idx],  self.data[idx], self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return index, img, target
    

class UnlabelData(torch.utils.data.Dataset): 
    def __init__(self, transform=None, target_transform=None, pos_data=None, \
            neg_data=None, index=None, data_type=None):
        self.transform = transform
        self.target_transform = target_transform

        self.data=np.concatenate((pos_data, neg_data), axis=0)
        self.true_targets = np.concatenate((np.zeros(pos_data.shape[0],  dtype= np.int_), np.ones(neg_data.shape[0],  dtype= np.int_)), axis=0)
        self.targets = np.ones_like(self.true_targets, dtype= np.int_)

        self.data_type = data_type
        self.index = index

    def __len__(self): 
        return len(self.targets)


    def __getitem__(self, idx):

        index, img, target, true_target = self.index[idx],  self.data[idx], self.targets[idx], self.true_targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return index, img, target, true_target

def get_PUDataSplits(data_obj, pos_size, alpha, beta, data_type=None): 

    unlabel_size = int((1-beta)*pos_size/beta)
    
    assert ((pos_size + int(unlabel_size*alpha)) <= len(data_obj.p_data)), "Check sizes again"
    assert ((int(unlabel_size*(1-alpha))) <= len(data_obj.n_data)), "Check sizes again"

    pos_data = data_obj.p_data[:pos_size]
    unlabel_pos_data = data_obj.p_data[pos_size: pos_size+ int(unlabel_size*alpha)]
    unlabel_neg_data = data_obj.n_data[:int(unlabel_size*(1-alpha))]

    pos, un =  PosData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                data=pos_data, index=np.array(range(pos_size)), data_type=data_type), \
            UnlabelData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                pos_data=unlabel_pos_data, neg_data=unlabel_neg_data, \
                index=np.array(range(unlabel_size)),data_type=data_type)
    return pos, un

def get_PNDataSplits(data_obj, pos_size, neg_size, data_type=None): 

    unlabel_pos_data = data_obj.p_data[:pos_size]
    unlabel_neg_data = data_obj.n_data[:neg_size]

    res = UnlabelData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                pos_data=unlabel_pos_data, neg_data=unlabel_neg_data, \
                index=np.array(range(pos_size + neg_size)),data_type=data_type)
    return res


def get_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size): 

    p_trainloader=None
    u_trainloader=None
    p_validloader=None
    u_validloader=None
    net=None
    X=None
    Y=None

    if data_type=="IMDb_BERT": 

        train_texts, train_labels = read_imdb_split(f'./{data_dir}/aclImdb/train')
        test_texts, test_labels = read_imdb_split(f'./{data_dir}/aclImdb/test')

        transform = initialize_bert_transform('distilbert-base-uncased')

        train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)

        p_traindata, u_traindata = get_PUDataSplits(train_dataset, pos_size=6250, alpha=alpha, beta=beta,data_type='IMDb_BERT')
        p_validdata, u_validdata = get_PUDataSplits(test_dataset, pos_size=5000, alpha=alpha, beta=beta,data_type='IMDb_BERT')

        X = p_traindata.targets
        Y = u_traindata.targets

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=8, \
            shuffle=True)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=8, \
            shuffle=True)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=128, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)

        ## Initialize model 
        net = get_model(net_type)
        net = net.to(device)

    elif data_type=="SemEval": 

        train_texts, train_labels = read_semeval_split(f'/share/garg/kkr36/Task_A', 'train')
        test_texts, test_labels = read_semeval_split(f'/share/garg/kkr36/Task_A', 'validation')

        np_train = sum(train_labels)
        nn_train = len(train_labels) - np_train
        np_test = sum(test_labels)
        nn_test = len(test_labels) - np_test
        
        # import pdb; pdb.set_trace()

        transform = initialize_codebert_transform("microsoft/codebert-base")

        train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)

        # neg = (1-beta) pos / beta
        # neg/pos = 1-beta / beta
        # neg / pos = 1/ beta - 1
        # 1 / beta = 1 + neg / pos
        # beta = 1 / (1 + neg / pos)
        alpha = 0
        beta_train = 1 / (1 + (nn_train / np_train))
        beta_test = 1 / (1 + (nn_test / np_test))

        p_traindata, u_traindata = get_PUDataSplits(train_dataset, pos_size=np_train, alpha=alpha, beta=beta_train,data_type='SemEval')
        p_validdata, u_validdata = get_PUDataSplits(test_dataset, pos_size=np_test, alpha=alpha, beta=beta_test,data_type='SemEval')

        X = p_traindata.targets
        Y = u_traindata.targets
        
        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=16, \
            shuffle=True)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=16, \
            shuffle=True)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=128, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)

        ## Initialize model 
        net = get_model(net_type)
        net = net.to(device)
    
    return p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata

    

def get_PN_dataset(data_dir, data_type,net_type, device,  alpha, beta, batch_size): 

    u_trainloader=None
    u_validloader=None
    net=None
    
    if data_type=="IMDb_BERT": 
        train_texts, train_labels = read_imdb_split(f'./{data_dir}/aclImdb/train')
        test_texts, test_labels = read_imdb_split(f'./{data_dir}/aclImdb/test')

        transform = initialize_bert_transform('distilbert-base-uncased')

        train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)


        u_traindata = get_PNDataSplits(train_dataset, pos_size=6250,  neg_size=int(6250*(1-alpha)*(1-beta)/beta) ,data_type='IMDb_BERT')
        u_validdata = get_PNDataSplits(test_dataset, pos_size=int(5000*alpha), neg_size=int(5000*(1-alpha)) ,data_type='IMDb_BERT')

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=128, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=256, \
            shuffle=True)

        ## Initialize model 
        net = get_model(net_type)
        net = net.to(device)

    elif data_type=="SemEval":
        # train_texts, train_labels = read_semeval_split(f'/share/garg/kkr36/Task_A/train.parquet', 'train')
        # test_texts, test_labels = read_semeval_split(f'/share/garg/kkr36/Task_A/train.parquet', 'validation')
        train_texts, train_labels = read_semeval_split(f'/share/garg/kkr36/Task_A', 'train')
        test_texts, test_labels = read_semeval_split(f'/share/garg/kkr36/Task_A', 'validation')

        np_train = sum(train_labels)
        nn_train = len(train_labels) - np_train
        np_test = sum(test_labels)
        nn_test = len(test_labels) - np_test

        transform = initialize_codebert_transform("microsoft/codebert-base")

        train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)
        # import pdb; pdb.set_trace()

        u_traindata = get_PNDataSplits(train_dataset, pos_size=np_train, neg_size=nn_train, data_type='SemEval')
        u_validdata = get_PNDataSplits(test_dataset, pos_size=np_test, neg_size=nn_test, data_type='SemEval')

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=16, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)

        ## Initialize model 
        net = get_model(net_type)
        net = net.to(device)

    return u_trainloader, u_validloader, net