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

class RCNN(nn.Module):
    def __init__(self,args):
        super(RCNN, self).__init__()
        article_dim = args['article_dim'] #500
        title_dim = args['title_dim'] #15
        label_dim = args['label_dim'] # 36321
        train_size = args['train_size']
        embed_dim = args['embed_dim'] # 300
        # self.embeding = nn.Embedding(vocb_size, dim,_weight=embedding_matrix)
        D = embed_dim
        C = label_dim
        self.num_layer = 1

        self.tdfc1 = nn.Linear(D, 300)
        self.td1 = TimeDistributed(self.tdfc1)
        self.tdbn1 = nn.BatchNorm2d(1)

        self.tdfc2 = nn.Linear(D, 300)
        self.td2 = TimeDistributed(self.tdfc2)
        self.tdbn2 = nn.BatchNorm2d(1)

        self.lstm1 = nn.LSTM(input_size = 300, hidden_size = 300, num_layers = self.num_layer, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size = 300, hidden_size = 300, num_layers = self.num_layer, batch_first=True, bidirectional=True)

        self.conv1 = nn.Conv2d(1, 1024 * 6, (3, 300+300+300))
        # self.conv1 = nn.Conv2d(1, 512, (1, 256+256+D))
        self.convbn1 = nn.BatchNorm2d(1024 * 6)
        # self.convbn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(1, 1024 * 6, (3, 300+300+300))
        # self.conv2 = nn.Conv2d(1, 512, (1, 256+256+D))
        self.convbn2 = nn.BatchNorm2d(1024 * 6)
        # self.convbn2 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(1024 * 6 * 2, C)
       	# self.fc = nn.Linear(1024, C)
        self.bn1 = nn.BatchNorm1d(C)#4096)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(C, C)#4096, C)   
        
    def forward(self, x, y):
        batch_size = x.size(0)
        x = x.detach()
        x = F.relu(self.tdbn1(self.td1(x).unsqueeze(1))).squeeze(1)
        y = y.detach()
        y = F.relu(self.tdbn2(self.td2(y).unsqueeze(1))).squeeze(1)

        h0_1 = torch.randn(self.num_layer * 2, batch_size, 300)
        c0_1 = torch.randn(self.num_layer * 2, batch_size, 300)
        # h0_1 = Variable(torch.randn(2, batch_size, self.D))
        # c0_1 = Variable(torch.randn(2, batch_size, self.D))
        h0_1 = h0_1.cuda()
        c0_1 = c0_1.cuda()
        o1, _ = self.lstm1(x, (h0_1, c0_1)) # (h0_1, c0_1) 
        x = torch.cat((x, o1), 2)
        x = F.relu(self.convbn1(self.conv1(x.unsqueeze(1))).squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        # y = self.get_context_embedding(y, 2)
        h0_2 = Variable(torch.randn(self.num_layer * 2, batch_size, 300))
        c0_2 = Variable(torch.randn(self.num_layer * 2, batch_size, 300))
        # h0_2 = Variable(torch.randn(2, batch_size, self.D))
        # c0_2 = Variable(torch.randn(2, batch_size, self.D))
        h0_2 = h0_2.cuda()
        c0_2 = c0_2.cuda()
        o2, _ = self.lstm2(y, (h0_2, c0_2))
        y = torch.cat((y, o2), 2)
        
        y = F.relu(self.convbn2(self.conv2(y.unsqueeze(1))).squeeze(3))
        y = F.max_pool1d(y, y.size(2)).squeeze(2)
        
        x = torch.cat((x, y), 1)
        x = x.view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        logit = self.fc2(x)
        
        return logit