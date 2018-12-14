import torch.utils.data as data
import torch
import pickle
from os import listdir
from os.path import join

class DatasetFromFile(data.Dataset):
    def __init__(self, fold_dir, input_transform=None, target_transform=True):
        super(DatasetFromFile, self).__init__()
        f = open(fold_dir + 'train_article_embedding_data.pickle','rb')
        self.train_article_embedding_data = pickle.load(f)
        f.close()
        f = open(fold_dir + 'train_title_embedding_data.pickle','rb')
        self.train_title_embedding_data = pickle.load(f)
        f.close()
        f = open(fold_dir + 'train_label_list_data.pickle','rb')
        self.train_label_list_data = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_article_embedding_data.pickle','rb')
        self.val_article_embedding_data = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_title_embedding_data.pickle','rb')
        self.val_title_embedding_data = pickle.load(f)
        f.close()
        f = open(fold_dir + 'val_label_list_data.pickle','rb')
        self.val_label_list_data = pickle.load(f)
        f.close()
        f = open(fold_dir + 'label_vocab.pickle','rb')
        self.label_vocab = pickle.load(f)
        f.close()
        self.input_transform = input_transform
        self.target_transform = target_transform

    def one_hot_label_embedding(self, label_list, label_vocab):
        tmp = label_list.copy()
        l = [0] * len(label_vocab)
        for j in tmp:
            if j in label_vocab:
                l[label_vocab[j]] = 1
        return l

    # 在__getitem__中加载图片，并且将传入的transformation操作运用到
    # 加载的图片中。 `input = self.input_transforms(input)`
    # 这里的 self.input_transforms就是传入的"类的实例"，由于类是callable的
    # 所以可以 "类的实例（参数）"这样调用。在上一篇博客说到了这个。
    def __getitem__(self, index):
        article_input = self.train_article_embedding_data[index]
        title_input = self.train_title_embedding_data[index]
        target_tags = self.train_label_list_data[index]
        if self.input_transform:
            article_input.extend(title_input)
        if self.target_transform:
            target_tags = self.one_hot_label_embedding(target_tags, self.label_vocab)
        input = torch.FloatTensor(article_input)
        label = torch.FloatTensor(target_tags)
        return input, label

    def __len__(self):
        return len(self.train_title_embedding_data)


tag_dataset = torch.utils.data.DataLoader(
         DatasetFromFile('../input/', input_transform=None, target_transform=True), 
         batch_size= 100, shuffle= False, num_workers= 4)