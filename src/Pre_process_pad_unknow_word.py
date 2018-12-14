import gensim 
import logging
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import random

# load data
output = open("train_list.pickle", "rb")
id_list, title_list, article_list = pickle.load(output)
output.close()

f = open('title_vocab.pickle','rb')
title_vocab = pickle.load(f)
f.close()

f = open('article_vocab.pickle','rb')
article_vocab = pickle.load(f)
f.close()


class Alphabet(dict):
    def __init__(self, start_id=1):
        self.fid = start_id
    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
            self.fid += 1
        return idx
    def dump(self, fname):
        with open(fname, 'w') as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

def parse_embed(embed_dict):
    alphabet, embed_mat = Alphabet(), []
    vocab_size, embed_dim = len(embed_dict), len(embed_dict[list(embed_dict.keys())[0]])
    unknown_word_idx = 0
    embed_mat.append(np.random.uniform(-0.25, 0.25, embed_dim))
    for word in embed_dict:
        alphabet.add(word)
        embed_mat.append(embed_dict[word])
    dummy_word_idx = alphabet.fid
    embed_mat.append(np.zeros(embed_dim))
    return alphabet, embed_mat, unknown_word_idx, dummy_word_idx

# unkown title_embed_mat[unknown_title_idx]
# pad title_embed_mat[dummy_title_idx]
# normal title_embed_mat[title_alphabet[i]]
title_alphabet, title_embed_mat, unknown_title_idx, dummy_title_idx = parse_embed(title_vocab)
article_alphabet, article_embed_mat, unknown_article_idx, dummy_article_idx = parse_embed(article_vocab)

tmp = np.array([len(i) for i in article_list])
# plt.hist(tmp)
tmp = np.array([len(i) for i in title_list])
# plt.hist(tmp)
len_aritle = 500 # mean 176.22896799025133
len_title = 15 # mean 8.178064471860113

def pad_list(alphabet, embed_mat, unknown_idx, dummy_idx, list_file, length):
    result = []
    for l in list_file:
        if len(l) > length:
            l = l[len(l) - length:]
        else:
            l = [0] * (length - len(l)) + l
        tmp = []
        for i in l:
            if i == 0:
                tmp.append(embed_mat[dummy_idx])
            elif i not in alphabet:
                tmp.append(embed_mat[unknown_idx])
            elif i in alphabet:
                tmp.append(embed_mat[alphabet[i]])
        result.append(tmp)
    return result

title_embedding_pad = pad_list(title_alphabet, title_embed_mat, unknown_title_idx, dummy_title_idx, title_list, len_title)

del id_list, title_list,title_alphabet, title_embed_mat, unknown_title_idx, dummy_title_idx, len_title,title_vocab

article_embedding_pad = pad_list(article_alphabet, article_embed_mat, unknown_article_idx, dummy_article_idx, article_list, len_aritle)

del article_list, article_alphabet, article_embed_mat, unknown_article_idx, dummy_article_idx, len_aritle, article_vocab

f = open('title_embedding_pad.pickle','wb')
pickle.dump(title_embedding_pad,f) 
f.close()

f = open('article_embedding_pad.pickle','wb')
pickle.dump(article_embedding_pad,f) 
f.close()





