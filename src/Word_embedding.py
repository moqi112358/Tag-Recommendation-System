import gensim 
import logging
import pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

def train(model, sentences, output_file='test.word2vec', train_sentences=None, epoch = 5):
    model.build_vocab(sentences)
    if train_sentences:
        model.train(train_sentences,total_examples=len(train_sentences),epochs=epoch)
    #model.save_word2vec_format(output_file)
    model.save(output_file)
    return model

def test_model_random(sentences, output_file):
    #model = Word2Vec.load_word2vec_format(output_file, binary=False)
    model = Word2Vec.load(output_file)
    list_sentences = list(sentences)
    for i in range(10):
        sentence = random.choice(list_sentences)
        while len(sentence) < 3:
            sentence = random.choice(list_sentences)
        word = random.choice(sentence)
        print(">>> %s: %s" % (word, " ".join(sentence)))
        try:
            for w,s in model.most_similar(word):
                print("%.6f %s" % (s, w))
        except:
            print("[WARN] low-frequency word")

def test_model(word_file, output_file):
    model = Word2Vec.load(output_file)
    print("# %s %s" % (model, output_file))
    for line in file(word_file):
        word = line.strip().decode('utf8')
        print(">>> %s" % (word))
        try:
            for w,s in model.most_similar(word):
                print("%.6f %s" % (s, w))
        except:
            print("[WARN] low-frequency word")

#def main():
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import random
CPU_COUNT = multiprocessing.cpu_count()

f = open("train_list.pickle", "rb")
id_list, title_list, article_list = pickle.load(f)
f.close()

model_title = gensim.models.Word2Vec(
       sg=1, # skip-gram,
       hs=0, #  negative sampling
       negative=10, # “noise words” 
       sample=1e-3,
       seed=0,
       size=300,
       window=2,
       min_count=2,
       workers=CPU_COUNT)

train(model_title, title_list, 'title_list.embedding', title_list, 5)
test_model_random(title_list, 'title_list.embedding')

model_article = gensim.models.Word2Vec(
       sg=1, # skip-gram
       hs=0, # negative sampling
       negative=5, # “noise words” 
       seed=0,
       size=300,
       sample=1e-5,
       window=5,
       min_count=5,
       workers=CPU_COUNT)

train(model_article, article_list, 'article_list.embedding', article_list, 5)
test_model_random(article_list, 'article_list.embedding')
# model_article.similarity('web', 'website')
# model_article = gensim.models.Word2Vec.load('article_list.embedding')

vocab_title = {i:np.array(model_title.wv[i]) for i in model_title.wv.index2word}
vocab_article = {i:np.array(model_article.wv[i]) for i in model_article.wv.index2word}

output = open("title_vocab.pickle", "wb")
pickle.dump(vocab_title, output)
output.close()

output = open("article_vocab.pickle", "wb")
pickle.dump(vocab_article, output)
output.close()