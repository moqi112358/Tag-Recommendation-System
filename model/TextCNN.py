import numpy as np # linear algebra
import pandas as pd
import os
import torch.utils.data as data
import torch
import pickle
from os import listdir
from os.path import join
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

def pre_read(fold_dir):
        args = {}
        f = open(fold_dir + 'train_article_embedding_data.pickle','rb')
        args['train_article_embedding_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'train_title_embedding_data.pickle','rb')
        args['train_title_embedding_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'train_label_list_data.pickle','rb')
        args['train_label_list_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_article_embedding_data.pickle','rb')
        args['val_article_embedding_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_title_embedding_data.pickle','rb')
        args['val_title_embedding_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_label_list_data.pickle','rb')
        args['val_label_list_data'] = pickle.load(f)
        f.close()
        f = open(fold_dir + 'label_vocab.pickle','rb')
        args['label_vocab'] = pickle.load(f)
        f.close()
        args['article_dim'] = len(args['train_article_embedding_data'][0])
        args['title_dim'] = len(args['train_title_embedding_data'][0])
        args['label_dim'] = len(args['label_vocab'])
        args['train_size'] = len(args['train_article_embedding_data'])
        args['embed_dim'] = len(args['train_article_embedding_data'][0][0])
        return args

args = pre_read('../input/train-validation-data-split/')

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        n, t = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # we have to reshape Y
        y = y.contiguous().view(n, t, y.size()[1])
        return y

class K_MaxPooling(nn.Module):
    def __init__(self, k):
        super(K_MaxPooling, self).__init__()
        self.k = k
        
    def forward(self, x, dim):
        index = x.topk(self.k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

class textCNN(nn.Module):
    def __init__(self,args):
        super(textCNN, self).__init__()
        article_dim = args['article_dim'] #500
        title_dim = args['title_dim'] #15
        label_dim = args['label_dim'] # 36321
        train_size = args['train_size']
        embed_dim = args['embed_dim'] # 300
        # self.embeding = nn.Embedding(vocb_size, dim,_weight=embedding_matrix)

        D = embed_dim
        C = label_dim
        Ci = 1
        Co = 300#opt['kernel_num']
        #Ks = opt['kernel_sizes']
        Ks1 = [1,2,3,4,5]
        Ks2 = [3,4,5,6,7]
        
        self.tdfc1 = nn.Linear(D, 300)#512)
        self.td1 = TimeDistributed(self.tdfc1)
        self.tdbn1 = nn.BatchNorm2d(1)
        
        self.tdfc2 = nn.Linear(D, 300)#512)
        self.td2 = TimeDistributed(self.tdfc2)
        self.tdbn2 = nn.BatchNorm2d(1)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, 290)) for K in Ks1])
        self.convbn1 = nn.ModuleList([nn.BatchNorm2d(Co) for i in range(len(Ks1))])
        self.convs2 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, 290)) for K in Ks2])
        self.convbn2 = nn.ModuleList([nn.BatchNorm2d(Co) for i in range(len(Ks2))])
        
        self.kmax_pooling = K_MaxPooling(5)
    
        self.fc1 = nn.Linear(33000, C)#4096)
        # self.fc1 = nn.Linear(115200, C)#4096)
        self.bn1 = nn.BatchNorm1d(C)#4096)
        self.fc2 = nn.Linear(C, C)#4096, C)

    def forward(self, x, y):
        batch_size = x.size(0)
        x = x.detach()
        x = F.relu(self.tdbn1(self.td1(x).unsqueeze(1)))
        y = y.detach()
        y = F.relu(self.tdbn2(self.td2(y).unsqueeze(1)))
        x = [F.relu(self.convbn1[i](conv(x))).squeeze(3) for i, conv in enumerate(self.convs1)]
        #print(x)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = [self.kmax_pooling(i, 2).mean(2).squeeze(1) for i in x]
        x = torch.cat(x, 1)
        
        y = [F.relu(self.convbn2[i](conv(y))).squeeze(3) for i, conv in enumerate(self.convs2)]
        # y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]
        y = [self.kmax_pooling(i, 2).mean(2).squeeze(1) for i in y]
        
        y = torch.cat(y, 1)
        x = torch.cat((x, y), 1)
        x = x.view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        logit = self.fc2(x)
        return logit