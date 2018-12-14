def read_vocab(fold_dir):        
    f = open(fold_dir + 'label_vocab.pickle','rb')
    args['label_vocab'] = pickle.load(f)
    f.close()
    f = open(fold_dir + 'title_vocab.pickle','rb')
    args['title_vocab'] = pickle.load(f)
    f.close()
    f = open(fold_dir + 'article_vocab.pickle','rb')
    args['article_vocab'] = pickle.load(f)
    f.close()
    return args

args = read_vocab('../input/tag-recommendation-system-list/')

f = open('../input/tag-recommendation-system-list/' + 'article_embedding_pad.pickle','rb')
article_embedding = pickle.load(f)
f.close()
f = open('../input/tag-recommendation-system-list/' + 'title_embedding_pad.pickle','rb')
title_embedding = pickle.load(f)
f.close()
miss_artitle_id = 0
#len = 247
miss_artitle = ['is', 'there', 'way', 'to', 'find', 'what', 'sort', 'of', 'requests', 'does', 'flash', 'application', 'send', 'to', 'server', 'was', 'trying', 'to', 'see', 'what', 'information', 'client', 'sends', 'to', 'the', 'server', 'using', 'chrome', 'inspect', 'element', 'but', 'it', 'shows', 'me', 'that', 'nothing', 'is', 'going', 'on', 'but', 'for', 'sure', 'the', 'communication', 'is', 'going', 'on', 'the', 'website', 'am', 'interesting', 'in', 'is', 'href', 'http', 'www', 'chesscube', 'com', 'rel', 'nofollow', 'http', 'www', 'chesscube', 'com', 'and', 'every', 'time', 'you', 'make', 'move', 'it', 'somehow', 'sends', 'it', 'to', 'server', 'or', 'may', 'be', 'just', 'to', 'another', 'opponent', 'in', 'the', 'end', 'of', 'the', 'game', 'it', 'sends', 'the', 'game', 'to', 'the', 'server', 'for', 'sure', 'but', 'up', 'till', 'now', 'all', 'can', 'see', 'is', 'just', 'few', 'images', 'being', 'uploaded', 'during', 'the', 'game', 'thanks', 'for', 'help', 'was', 'trying', 'to', 'use', 'wireshark', 'to', 'capture', 'packets', 'and', 'in', 'such', 'way', 'to', 'see', 'the', 'communication', 'here', 'what', 'was', 'doing', 'pinging', 'chesscube', 'com', 'to', 'realize', 'its', 'ip', 'address', 'than', 'am', 'listening', 'only', 'for', 'packages', 'from', 'that', 'ip', 'address', 'ip', 'addr', 'but', 'the', 'only', 'thing', 'can', 'see', 'is', 'lot', 'of', 'tcp', 'and', 'some', 'http', 'packages', 'all', 'http', 'packages', 'are', 'sending', 'just', 'png', 'images', 'of', 'the', 'avatars', 'of', 'the', 'users', 'there', 'is', 'chat', 'there', 'and', 'people', 'are', 'constantly', 'speaking', 'but', 'can', 'no', 'see', 'that', 'understand', 'that', 'it', 'is', 'going', 'from', 'another', 'ip', 'address', 'but', 'have', 'no', 'idea', 'how', 'can', 'found', 'out', 'it', 'the', 'problem', 'is', 'that', 'can', 'not', 'watch', 'for', 'all', 'traffic', 'between', 'the', 'net', 'and', 'my', 'computer', 'because', 'there', 'is', 'so', 'much', 'of', 'it', 'and', 'do', 'not', 'know', 'how', 'limit', 'it']
miss_artitle_word = 'chesscube'
miss_artitle.index(miss_artitle_word) # 56
miss_artitle_id_2 = 4
miss_artitle_2 = ['are', 'there', 'any', 'rpc', 'modules', 'which', 'work', 'with', 'promises', 'on', 'the', 'server', 'have', 'functions', 'which', 'return', 'promises', 'would', 'like', 'to', 'expose', 'them', 'for', 'browser', 'clients', 'to', 'call', 'over', 'websockts', 'or', 'fallbacks', 'found', 'some', 'rpc', 'libraries', 'for', 'example', 'dnode', 'but', 'they', 'expect', 'callback', 'as', 'parameter', 'would', 'like', 'something', 'like', 'this', 'server', 'pre', 'rpc', 'expose', 'timeout', 'function', 'time', 'var', 'defer', 'settimeout', 'function', 'resolve', 'time', 'return', 'promise', 'pre', 'client', 'pre', 'rpc', 'timeout', 'then', 'function', 'console', 'log', 'done', 'pre']
miss_artitle_word_2 = 'websockts'
miss_artitle_2.index(miss_artitle_word_2) # 28
miss_title_id = 20
miss_title = ['segmentation', 'fault', 'cmovl']
miss_title_word = 'cmovl'
miss_title.index(miss_title_word) # 2
miss_title_id_2 = 23
miss_title_2 = ['how', 'to', 'set', 'color', 'with', 'leaflet', 'glify']
miss_title_word_2 = 'glify'
miss_title_2.index(miss_title_word_2) # 6
a = article_embedding[miss_artitle_id_2][500 - len(miss_artitle_2) + miss_artitle_2.index(miss_artitle_word_2) + 0]
b = article_embedding[miss_artitle_id][500 - len(miss_artitle) + miss_artitle.index(miss_artitle_word) + 0]
unknow_for_article_embedding = a
a =title_embedding[miss_title_id_2][15 - len(miss_title_2) + miss_title_2.index(miss_title_word_2) + 0]
b = title_embedding[miss_title_id][15 - len(miss_title) + miss_title.index(miss_title_word) + 0]
unknow_for_title_embedding = a

f = open('unknow_word_embedding.pickle','wb')
pickle.dump([unknow_for_article_embedding,unknow_for_title_embedding],f)
f.close()