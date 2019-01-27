from preprocessing.preprocessing import preprocessing
from preprocessing.training import train

word2embed, word2id, id2word, seq2words, labels, valid_data, test_data = preprocessing()
train(word2embed, word2id, id2word, seq2words, labels, valid_data)
