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
import os

from data_helper import * 
from model_helper import * 
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.enable_pipe("senter")

def split_into_sentences(abstracts, labels):
    all_sentences = []
    all_labels = []
    print("Splitting into sentences!")
    for i, abstract in tqdm(list(enumerate(abstracts))):
        doc = nlp(abstract)
        sentences = [sent.text.strip() for sent in doc.sents]
        all_sentences.extend(sentences)  # extend instead of append
        all_labels.extend([labels[i] for _ in range(len(sentences))])
    return all_sentences, all_labels

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
        
        if self.data_type == 'cifar' : 
            img = Image.fromarray(img)
        
        elif self.data_type =='mnist': 
            img = Image.fromarray(img, mode='L')

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
        
        if self.data_type == 'cifar' : 
            img = Image.fromarray(img)
        
        elif self.data_type =='mnist': 
            img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return index, img, target, true_target

def get_PUDataSplits(data_obj, pos_size, alpha, beta, data_type=None): 

    unlabel_size = int((1-beta)*pos_size/beta)
    # import pdb; pdb.set_trace()
    
    assert ((pos_size + int(unlabel_size*alpha)) <= len(data_obj.p_data)), "Check sizes again"
    assert ((int(unlabel_size*(1-alpha))) <= len(data_obj.n_data)), "Check sizes again"

    pos_data = data_obj.p_data[:pos_size]
    unlabel_pos_data = data_obj.p_data[pos_size: pos_size+ int(unlabel_size*alpha)]
    unlabel_neg_data = data_obj.n_data[:int(unlabel_size*(1-alpha))]

    # if alpha == 0: import pdb; pdb.set_trace()

    return PosData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                data=pos_data, index=np.array(range(pos_size)), data_type=data_type), \
            UnlabelData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                pos_data=unlabel_pos_data, neg_data=unlabel_neg_data, \
                index=np.array(range(unlabel_size)),data_type=data_type)

def get_PUDataSplits1(data_obj, data_type=None): # put all the mirrors in posdata; put all arxiv abstracts in unlabeled data as negatives

    return  PosData(transform=data_obj.transform, \
                    target_transform=data_obj.target_transform, \
                    data=data_obj.p_data, index=np.array(range(len(data_obj.p_data))), data_type=data_type), \
            UnlabelData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                pos_data=data_obj.p_data[:0], neg_data=data_obj.n_data, \
                index=np.array(range(len(data_obj.n_data))),data_type=data_type)

def get_PUDataSplits2(data_obj, pos_size, alpha, beta, data_type=None): 

    unlabel_size = int((1-beta)*pos_size/beta)
    
    assert ((pos_size + int(unlabel_size*alpha)) <= len(data_obj.p_data)), "Check sizes again"
    assert ((int(unlabel_size*(1-alpha))) <= len(data_obj.n_data)), "Check sizes again"

    pos_data = data_obj.p_data[:pos_size]
    unlabel_pos_data = data_obj.p_data[pos_size: pos_size+ int(unlabel_size*alpha)]
    unlabel_pos_data = np.vstack([pos_data, unlabel_pos_data])
    unlabel_neg_data = data_obj.n_data[:int(unlabel_size*(1-alpha))]

    # if alpha == 0: import pdb; pdb.set_trace()

    return UnlabelData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                pos_data=unlabel_pos_data, neg_data=unlabel_neg_data, \
                index=np.array(range(unlabel_size + pos_size)),data_type=data_type)

def get_PNDataSplits(data_obj, pos_size, neg_size, data_type=None): 

    unlabel_pos_data = data_obj.p_data[:pos_size]
    unlabel_neg_data = data_obj.n_data[:neg_size]

    return UnlabelData(transform=data_obj.transform, \
                target_transform=data_obj.target_transform, \
                pos_data=unlabel_pos_data, neg_data=unlabel_neg_data, \
                index=np.array(range(pos_size + neg_size)),data_type=data_type)


def get_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size, year, sentence): 

    p_trainloader=None
    u_trainloader=None
    p_validloader=None
    u_validloader=None
    net=None
    X=None
    Y=None

    if data_type=='gaussian': 
        '''
        Gaussian Data hyperparamters and data
        '''
        num_points = 6000
        input_size =200
        pos_size = 2000

        gauss_traindata = Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=num_points, dim=input_size//2)
        gauss_testdata = Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=num_points, dim=input_size//2)

        p_traindata, u_traindata = get_PUDataSplits(gauss_traindata, pos_size=pos_size, alpha=alpha, beta=beta)
        p_validdata , u_validdata = get_PUDataSplits(gauss_testdata, pos_size=pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 

        net = get_model(net_type, input_dim = input_size)
        net = net.to(device)

    elif data_type=='toy_continuous':       
        '''
        Toy dataset from P vs U failure for domain discrimination
        '''

        toy_traindata = ToyDataContinuous()
        toy_testdata = ToyDataContinuous()

        pos_size = 50
        p_traindata, u_traindata = get_PUDataSplits(toy_traindata, pos_size=pos_size, alpha=alpha, beta=beta)
        p_validdata, u_validdata = get_PUDataSplits(toy_testdata, pos_size=pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 2)
        net = net.to(device)

    elif data_type=='toy_discrete': 

        toy_traindata = ToyData()
        toy_testdata = ToyData()

        pos_size = 8
        p_traindata, u_traindata = get_PUDataSplits(toy_traindata, pos_size=pos_size, alpha=alpha, beta=beta)
        p_validdata, u_validdata = get_PUDataSplits(toy_testdata, pos_size=pos_size, alpha=alpha, beta=beta)

        X = p_traindata.data
        Y = u_traindata.data

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 2)
        net = net.to(device)


    elif data_type=='mnist_17': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = MNIST17Data(root=data_dir, train=True, transform=transform_train)
        testdata = MNIST17Data(root=data_dir, train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=3000, alpha=alpha, beta=beta,data_type='mnist')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=500, alpha=alpha, beta=beta,data_type='mnist')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)


    elif data_type=='mnist_binarized': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = BinarizedMNISTData(root=data_dir, train=True, transform=transform_train)
        testdata = BinarizedMNISTData(root=data_dir, train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=15000, alpha=alpha, beta=beta,data_type='mnist')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=2500, alpha=alpha, beta=beta,data_type='mnist')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)

    elif data_type=='mnist_overlap': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = OverlapMNISTData(root=data_dir, train=True, transform=transform_train)
        testdata = OverlapMNISTData(root=data_dir, train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=15000, alpha=alpha, beta=beta,data_type='mnist')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=2500, alpha=alpha, beta=beta,data_type='mnist')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)

    elif data_type=='cifar_DogCat': 

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        traindata = DogCatData(root=data_dir, train=True, transform=transform_train)
        testdata = DogCatData(root=data_dir, train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=2500, alpha=alpha, beta=beta,data_type='cifar')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=500, alpha=alpha, beta=beta,data_type='cifar')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 3072)
        net = net.to(device)
    
    elif data_type=='cifar_binarized': 

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        traindata = BinarizedCifarData(root=data_dir, train=True, transform=transform_train)
        testdata = BinarizedCifarData(root=data_dir, train=False, transform=transform_test)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=12500, alpha=alpha, beta=beta,data_type='cifar')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=2500, alpha=alpha, beta=beta,data_type='cifar')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 3072)
        net = net.to(device)

    elif data_type.startswith("UCI"): 
        
        uci_data_type = data_type.split("_")[1]
        p_data, n_data = uci_data(uci_data_type)

        keep_num = min(len(p_data), len(n_data))
        p_data = p_data[:keep_num]
        n_data = n_data[:keep_num]

        train_p_size= keep_num//3
        test_p_size= keep_num//6


        traindata = UCI_data(p_data=p_data, n_data=n_data, train=True)
        testdata = UCI_data(p_data=p_data, n_data=n_data, train=False)

        p_traindata, u_traindata = get_PUDataSplits(traindata, pos_size=train_p_size, alpha=alpha, beta=beta,data_type='uci')
        p_validdata, u_validdata = get_PUDataSplits(testdata, pos_size=test_p_size, alpha=alpha, beta=beta,data_type='uci')

        X = p_traindata.data.reshape((p_traindata.data.shape[0], -1))
        Y = u_traindata.data.reshape((u_traindata.data.shape[0], -1))

        p_trainloader = torch.utils.data.DataLoader(p_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)
        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=int(batch_size*(1-beta)/beta), \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = p_data.shape[-1])
        net = net.to(device)

    elif data_type=="IMDb_BERT": 

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
    
    elif data_type=="ArXiv_BERT": 
        # train_path = f'{data_dir}/alpha/train/arxiv_tokenized_{year}_cs._2000.parquet' # TODO figure out year
        # val_path = f'{data_dir}/alpha/val/arxiv_tokenized_{year}_cs._500.parquet'

        data_path = f'{data_dir}/multillm/data_raw/arxiv_{year}_ai_cs._10000_fronthalf.parquet'

        train_texts, train_labels = read_arxiv_split2(data_path, "train") # should have 15k each
        test_texts, test_labels = read_arxiv_split2(data_path, "val") # should have 5k each
        # import pdb; pdb.set_trace()

        if sentence:
            train_texts, train_labels = split_into_sentences(train_texts, train_labels)
            test_texts, test_labels = split_into_sentences(test_texts, test_labels)
        # train_texts, train_labels = read_arxiv_split(train_path) # should have 15k each
        # test_texts, test_labels = read_arxiv_split(val_path) # should have 5k each

        # train_texts = [' '.join(text) for text in train_texts]
        # test_texts = [' '.join(text) for text in test_texts]
        # import pdb; pdb.set_trace()

        transform = initialize_bert_transform('distilbert-base-uncased')

        train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)

        np_train, nn_train = len(train_dataset.p_data), len(train_dataset.n_data)
        np_test, nn_test = len(test_dataset.p_data), len(test_dataset.n_data)

        # p_traindata, u_traindata = get_PUDataSplits(train_dataset, pos_size=int(np_train*(1-alpha)), alpha=alpha, beta=(1-alpha)/(2-alpha),data_type=data_type)
        # p_validdata, u_validdata = get_PUDataSplits(test_dataset, pos_size=int(np_test*(1-alpha)), alpha=alpha, beta=(1-alpha)/(2-alpha),data_type=data_type)
        p_traindata, u_traindata = get_PUDataSplits1(train_dataset, data_type=data_type)
        p_validdata, u_validdata = get_PUDataSplits1(test_dataset, data_type=data_type)

        # import pdb; pdb.set_trace()

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

    return p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata

def get_dataset_val(data_dir, data_type,net_type, device, alpha, beta, batch_size, year, sentence): # TODO fix
    p_validloader=None
    u_validloader=None
    
    if data_type=="ArXiv_BERT": # lol this is all we support for now
        val_path = f'{data_dir}/alpha/val/arxiv_tokenized_{year}_cs._500.parquet'

        test_texts, test_labels = read_arxiv_split(val_path)
        test_texts = [' '.join(text) for text in test_texts]

        transform = initialize_bert_transform('distilbert-base-uncased')

        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)

        # import pdb; pdb.set_trace()

        np_test, nn_test = len(test_dataset.p_data), len(test_dataset.n_data)
        # import pdb; pdb.set_trace()

        p_validdata, u_validdata = get_PUDataSplits(test_dataset, pos_size=int(np_test*.5), alpha=alpha, beta=beta,data_type=data_type)

        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=128, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)

    # import pdb; pdb.set_trace()

    return p_validloader, u_validloader, p_validdata, u_validdata

def get_dataset_val2(data_dir, data_type,net_type, device, alpha, beta, batch_size, year, sentence): # TODO fix
    p_validloader=None
    u_validloader=None
    
    if data_type=="ArXiv_BERT": # lol this is all we support for now
        # val_path = f'{data_dir}/multillm/data_raw/arxiv_{year}_ai_cs._20000.parquet'

        val_path = f'{data_dir}/multillm/data_raw/arxiv_{year}_ai_cs._10000_fronthalf.parquet'

        test_texts, test_labels = read_arxiv_split2(val_path, "val")
        if sentence:
            test_sentences, new_test_labels = split_into_sentences(test_texts, test_labels)
            # import pdb; pdb.set_trace()
            test_texts, test_labels = test_sentences, new_test_labels
        # import pdb; pdb.set_trace()
        # test_texts = [' '.join(text) for text in test_texts]

        transform = initialize_bert_transform('distilbert-base-uncased')

        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)
        # import pdb; pdb.set_trace()

        np_test, nn_test = len(test_dataset.p_data), len(test_dataset.n_data)
        # assert(np_test == 5000 and nn_test == 5000)
        if not sentence:
            p_validdata, u_validdata = get_PUDataSplits(test_dataset, pos_size=int(np_test - alpha*np_test), alpha=alpha, beta=(1-alpha)/(2-alpha),data_type=data_type)
        else:
            p_validdata, u_validdata = get_PUDataSplits(test_dataset, pos_size=int(np_test - alpha*np_test), alpha=alpha, beta=(np_test / (nn_test + np_test)),data_type=data_type)
        # import pdb; pdb.set_trace()

        p_validloader = torch.utils.data.DataLoader(p_validdata, batch_size=128, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)

    # import pdb; pdb.set_trace()

    return p_validloader, u_validloader, p_validdata, u_validdata    

def get_PN_dataset(data_dir, data_type,net_type, device,  alpha, beta, batch_size, year, sentence): 

    u_trainloader=None
    u_validloader=None
    net=None

    if data_type=='gaussian': 
        '''
        Gaussian Data hyperparamters and data
        '''
        num_points = 6000
        input_size =200
        pos_size = 2000

        gauss_traindata = Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=num_points, dim=input_size//2)
        gauss_testdata = Gaussain_data(mu=1.0, sigma=np.sqrt(input_size//2), size=num_points, dim=input_size//2)

        u_traindata = get_PNDataSplits(gauss_traindata, unlabeled_size=int(num_points//2))
        u_validdata = get_PNDataSplits(gauss_traindata, unlabeled_size=int(num_points//2))


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 

        net = get_model(net_type, input_dim = input_size)
        net = net.to(device)

    elif data_type=='toy_continuous':       
        '''
        Toy dataset from P vs U failure for domain discrimination
        '''

        toy_traindata = ToyDataContinuous()
        toy_testdata = ToyDataContinuous()

        pos_size = 50
        u_traindata = get_PNDataSplits(toy_traindata, unlabeled_size=pos_size*2)
        u_validdata = get_PNDataSplits(toy_testdata, unlabeled_size=pos_size*2)

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size*2, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size*2, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 2)
        net = net.to(device)

    elif data_type=='toy_discrete': 

        toy_traindata = ToyData()
        toy_testdata = ToyData()

        pos_size = 8
        u_traindata =  get_PNDataSplits(toy_traindata, unlabeled_size=pos_size)
        u_validdata = get_PNDataSplits(toy_testdata, unlabeled_size=pos_size)


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=pos_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=pos_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 2)
        net = net.to(device)


    elif data_type=='mnist_17': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = MNIST17Data(root=data_dir, train=True, transform=transform_train)
        testdata = MNIST17Data(root=data_dir, train=False, transform=transform_test)

        u_traindata = get_PNDataSplits(traindata, pos_size=3000, neg_size=int(3000*(1-alpha)*(1-beta)/beta), data_type='mnist')
        u_validdata = get_PNDataSplits(testdata,pos_size=int(500*alpha), neg_size=int(500*(1-alpha)),data_type='mnist')


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)


    elif data_type=='mnist_binarized': 

        transform_train = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        transform_test = transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

        traindata = BinarizedMNISTData(root=data_dir, train=True, transform=transform_train)
        testdata = BinarizedMNISTData(root=data_dir, train=False, transform=transform_test)

        u_traindata = get_PNDataSplits(traindata, pos_size=15000, neg_size=int(15000*(1-alpha)*(1-beta)/beta), data_type='mnist')
        u_validdata = get_PNDataSplits(testdata, pos_size=int(2500*alpha), neg_size=int(2500*(1 - alpha)), data_type='mnist')


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 784)
        net = net.to(device)

    elif data_type=='cifar_DogCat': 

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        traindata = DogCatData(root=data_dir, train=True, transform=transform_train)
        testdata = DogCatData(root=data_dir, train=False, transform=transform_test)

        u_traindata = get_PNDataSplits(traindata, pos_size=2500, neg_size=int(2500*(1-alpha)*(1-beta)/beta), data_type='cifar')
        u_validdata = get_PNDataSplits(testdata, pos_size=int(500*alpha), neg_size=int(500*alpha), data_type='cifar')

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 3072)
        net = net.to(device)
    
    elif data_type=='cifar_binarized': 

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        traindata = BinarizedCifarData(root=data_dir, train=True, transform=transform_train)
        testdata = BinarizedCifarData(root=data_dir, train=False, transform=transform_test)

        u_traindata = get_PNDataSplits(traindata,pos_size=12500, neg_size=int(12500*(1-alpha)*(1-beta)/beta),data_type='cifar')
        u_validdata = get_PNDataSplits(testdata,pos_size=int(2500*alpha), neg_size=int(2500*(1-alpha)),data_type='cifar')


        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=batch_size, \
            shuffle=True, num_workers=2)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=batch_size, \
            shuffle=True, num_workers=2)

        ## Initialize model 
        net = get_model(net_type, input_dim = 3072)
        net = net.to(device)
    
    elif data_type=="IMDb_BERT": 
        train_texts, train_labels = read_imdb_split(f'./{data_dir}/aclImdb/train')
        test_texts, test_labels = read_imdb_split(f'./{data_dir}/aclImdb/test')

        transform = initialize_bert_transform('distilbert-base-uncased')

        train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)


        u_traindata = get_PNDataSplits(train_dataset, pos_size=6250,  neg_size=int(6250*(1-alpha)*(1-beta)/beta) ,data_type='IMDb_BERT')
        u_validdata = get_PNDataSplits(test_dataset, pos_size=int(5000*alpha), neg_size=int(5000*(1-alpha)) ,data_type='IMDb_BERT')

        # X = p_traindata.targets
        # Y = u_traindata.targets

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=8, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)

        ## Initialize model 
        net = get_model(net_type)
        net = net.to(device)

    elif data_type=="ArXiv_BERT": 
        # train_path = f'{data_dir}/alpha/train/arxiv_tokenized_{year}_cs._2000.parquet'
        # val_path = f'{data_dir}/alpha/val/arxiv_tokenized_{year}_cs._500.parquet'
        data_path = f'{data_dir}/multillm/data_raw/arxiv_{year}_ai_cs.10000_fronthalf.parquet'

        train_texts, train_labels = read_arxiv_split2(data_path, "train")
        test_texts, test_labels = read_arxiv_split2(data_path, "val")
        if sentence:
            train_texts, train_labels = split_into_sentences(train_texts, train_labels)
            test_texts, test_labels = split_into_sentences(test_texts, test_labels)

        # train_texts, train_labels = read_arxiv_split(train_path)
        # test_texts, test_labels = read_arxiv_split(val_path)
        # train_texts = [' '.join(text) for text in train_texts]
        # test_texts = [' '.join(text) for text in test_texts]

        # import pdb; pdb.set_trace()

        transform = initialize_bert_transform('distilbert-base-uncased')

        train_dataset = IMDbBERTData(train_texts, train_labels, transform=transform)
        test_dataset = IMDbBERTData(test_texts, test_labels, transform=transform)

        np_train, nn_train = len(train_dataset.p_data), len(train_dataset.n_data)
        np_test, nn_test = len(test_dataset.p_data), len(test_dataset.n_data)
        
        # u_traindata = get_PUDataSplits2(train_dataset, pos_size=int(np_train*(1-alpha)), alpha=alpha, beta=(1-alpha)/(2-alpha),data_type=data_type)
        # u_validdata = get_PUDataSplits2(test_dataset, pos_size=int(np_test*(1-alpha)), alpha=alpha, beta=(1-alpha)/(2-alpha),data_type=data_type)
        u_traindata = get_PNDataSplits(train_dataset, pos_size = np_train, neg_size = nn_train)
        u_validdata = get_PNDataSplits(train_dataset, pos_size = np_test, neg_size = nn_test)

        u_trainloader = torch.utils.data.DataLoader(u_traindata, batch_size=8, \
            shuffle=True)
        u_validloader = torch.utils.data.DataLoader(u_validdata, batch_size=128, \
            shuffle=True)
        # import pdb; pdb.set_trace()

        ## Initialize model 
        net = get_model(net_type)
        net = net.to(device)

    return u_trainloader, u_validloader, net