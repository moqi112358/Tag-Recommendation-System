import pickle
import random
# 5000 train - 3G
# 5000 label
def train_val_split(file_dir):
    # read data
    f = open(file_dir + 'article_embedding_pad.pickle','rb')
    article_embedding = pickle.load(f)
    f.close()
    f = open(file_dir + 'title_embedding_pad.pickle','rb')
    title_embedding = pickle.load(f)
    f.close()
    f = open(file_dir + 'train_label_list.pickle','rb')
    label_list = pickle.load(f)
    f.close()
    # split validation 5000 samples
    article_id = [i for i in range(len(article_embedding))]
    random.seed(0)
    random.shuffle(article_id)
    val_id = article_id[:5000]
    train_id = article_id[5000:]
    train_article_embedding_data = [article_embedding[i] for i in train_id]
    train_title_embedding_data = [title_embedding[i] for i in train_id]
    train_label_list_data = [label_list[i] for i in train_id]
    val_article_embedding_data = [article_embedding[i] for i in val_id]
    val_title_embedding_data = [title_embedding[i] for i in val_id]
    val_label_list_data = [label_list[i] for i in val_id]
    # create vocal label 
    tmp_label1 = train_label_list_data.copy()
    tmp_label2 = val_label_list_data.copy()
    tmp_label1.extend(tmp_label2)
    tmp = {}
    for word in tmp_label1:
        for i in word:
            tmp[i] = tmp.get(i,0) + 1
    label_vocab = {}
    index = 0
    for key in tmp:
        if tmp[key] > 10: # maintain label which appears for more than 10 times
            label_vocab[key] = index
            index += 1
    return train_article_embedding_data, train_title_embedding_data, train_label_list_data,val_article_embedding_data, val_title_embedding_data, val_label_list_data, label_vocab

train_article_embedding_data, train_title_embedding_data, train_label_list_data,\
val_article_embedding_data, val_title_embedding_data, val_label_list_data, label_vocab = train_val_split('../input/')

f = open('train_article_embedding_data.pickle','wb')
pickle.dump(f)
f.close()
f = open('train_title_embedding_data.pickle','wb')
pickle.dump(f)
f.close()
f = open('train_label_list_data.pickle','wb')
pickle.dump(f)
f.close()
f = open('val_article_embedding_data.pickle','wb')
pickle.dump(f)
f.close()
f = open('val_title_embedding_data.pickle','wb')
pickle.dump(f)
f.close()
f = open('val_label_list_data.pickle','wb')
pickle.dump(f)
f.close()
