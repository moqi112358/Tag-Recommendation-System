import os
os.getcwd()
os.chdir('C://Users/98302/Desktop/hackerearth/new_dataset/raw_data/')
import pandas as pd
raw_data = pd.read_csv('train.csv')

import gensim 
import logging
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

def read_input(raw_data):
    logging.info("reading raw data...this may take a while")
    l = len(raw_data)
    article_list = []
    title_list = []
    id_list = []
    for i in range(l):
        line = raw_data['article'].iloc[i]
        title = raw_data['title'].iloc[i]
        id = raw_data['id'].iloc[i]
        if (i % 10000 == 0):
            logging.info("read {0} reviews".format(i))
        # do some pre-processing and return list of words for each article
        line = gensim.utils.simple_preprocess(line)
        title = gensim.utils.simple_preprocess(title)
        article_list.append(line)
        title_list.append(title)
        id_list.append(id)
    return id_list, title_list, article_list

output = open("train_list.pickle", "wb")
id_list, title_list, article_list = read_input(raw_data)
pickle.dump([id_list, title_list, article_list], output)
output.close()

def read_label(raw_data):
    logging.info("reading raw data...this may take a while")
    l = len(raw_data)
    label_list = []
    for i in range(l):
        line = raw_data['tags'].iloc[i]
        line = str(line)
        if (i % 10000 == 0):
            logging.info("read {0} reviews".format(i))
        # do some pre-processing and return list of words for each article
        line = line.strip().split('|')
        label_list.append(line)
    return label_list

label_list = read_label(raw_data)
output = open("train_label_list.pickle", "wb")
pickle.dump(label_list, output)
output.close()

tmp = set()
for i in label_list:
    for j in i:
        tmp.add(j)
tmp = list(tmp)
label_vocab = {tmp[i]:i for i in range(len(tmp))}

output = open("label_vocab.pickle", "wb")
pickle.dump(label_vocab, output)
output.close()