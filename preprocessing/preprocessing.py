import logging
from collections import Counter

import nltk
import numpy as np
from torch import LongTensor
from torch.autograd import Variable

from preprocessing.embeddings import get_embeddings
from preprocessing.read_corpus import read_dataset


def preprocessing():
    train_data = read_dataset('data/train.txt')
    test_data = read_dataset('data/test.txt')
    valid_data = read_dataset('data/valid.txt')

    word2id, id2word, seq2words, train_labels = tokenize_text(train_data)
    test2id, id2test, testseq2words, _ = tokenize_text(test_data)
    valid2id, id2valid, validseq2words, _ = tokenize_text(valid_data)

    words = list(set(word2id.keys()) | set(test2id.keys()))
    words = list((set(words)) | set(valid2id.keys()))
    word2embed, nos_of_unk_words, nos_all_words = get_embeddings(words)

    print("Number of missing words is {}".format(nos_of_unk_words))
    print("Number of all words is {}".format(nos_all_words + nos_of_unk_words))

    return word2embed, word2id, id2word, seq2words, train_labels, valid_data, test_data


def to_lowercase(words):
    return [word.lower() for word in words]


def tokenize(sentence):
    words = nltk.word_tokenize(sentence)
    words = to_lowercase(words)
    return words


def tokenize_text_generator(gen):
    word2id, id2word = Counter(), Counter()
    seq2words = []
    while True:
        try:
            label, sent1, sent2 = next(gen)
            tokenized_sent1, tokenized_sent2 = tokenize(sent1), tokenize(sent2)
            for w in tokenized_sent1 + tokenized_sent2:
                if not w in word2id.keys():
                    word2id[w] = len(id2word)
                    id2word[word2id[w]] = w
            seq2words.append((tokenized_sent1, tokenized_sent2))
        except StopIteration:
            print("End of dataset")

    return word2id, id2word, seq2words


def tokenize_text(text):
    word2id, id2word = {}, {}
    seq2words, labels = [], []
    label_conversion = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    special_words = ['<pad>', '<bos>', '<eos>', '<unk>']

    for word in special_words:
        word2id[word] = len(id2word)
        id2word[word2id[word]] = word

    for i, triple in enumerate(text):
        tokenized_sent1, tokenized_sent2 = tokenize(triple[1]), tokenize(triple[2])
        for w in tokenized_sent1 + tokenized_sent2:
            if not w in word2id.keys():
                word2id[w] = len(id2word)
                id2word[word2id[w]] = w

        tokenized_sent1 = [id2word[1]] + tokenized_sent1 + [id2word[2]]
        tokenized_sent2 = [id2word[1]] + tokenized_sent2 + [id2word[2]]
        seq2words.append((tokenized_sent1, tokenized_sent2))
        labels.append(label_conversion[triple[0]])

        if i % 10000 == 0:
            print("round {}".format(i))

    return word2id, id2word, seq2words, labels


def find_longest_sentece(seq2words):
    return max(np.array([max(len(pair[0]), len(pair[1])) for pair in seq2words]))


def padding_filling(seq2words, max_len_sent):
    for sent in seq2words:
        sent.extend(['<pad>'] * (max_len_sent - len(sent)))
    return seq2words


def create_one_batch(batch, word2id, max_len_sent):
    batch_size = len(batch)
    padding_filling(batch, max_len_sent)

    unk_id = word2id.get('<unk>', None)
    batch_w = LongTensor(batch_size, max_len_sent)
    for i, batch_i in enumerate(batch):
        for j, batch_ij in enumerate(batch_i):
            batch_w[i][j] = word2id.get(batch_ij, unk_id)

    return batch_w


def create_batches(seq2words, labels, batch_size, word2id, max_len_sent):
    sum_len = 0.0
    batches_w, batches_labels = [], []
    size = batch_size
    print("Seq2words size : ", len(seq2words))
    batches = (len(seq2words) - 1) // size + 1
    print("Number of batches: ", batches)

    for i in range(batches):
        if i % 500 == 0:
            print("BATCH {}".format(i))
        start_id, end_id = i * size, (i + 1) * size
        decompress = list(map(list, zip(*seq2words[start_id: end_id])))
        batch_labels = labels[start_id: end_id]
        to_add = sum(len(sent) for sent in decompress[0] + decompress[1])

        pre_words = create_one_batch(decompress[0], word2id, max_len_sent)
        hypo_words = create_one_batch(decompress[1], word2id, max_len_sent)

        sum_len += to_add
        batches_w.append((pre_words, hypo_words))
        batches_labels.append(Variable(LongTensor(batch_labels)))

    logging.info("{} batches, avg len: {:.1f}".format(batches, sum_len / 2 * len(seq2words)))
    return batches_w, batches_labels
