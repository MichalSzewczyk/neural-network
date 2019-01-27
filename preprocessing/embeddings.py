import csv

import numpy as np
import pandas
from torch import FloatTensor


def get_embeddings(words_in_snli):
    special_words = ['<unk>', '<bos>', '<eos>', '<pad>']
    all_embeddings = parse_dataset('data/glove.6B.200d.txt')
    embeddings_matrix = {}
    embedding_size, nos_of_unk_words = 200, 0
    print(len(all_embeddings))

    for word in special_words:
        embeddings_matrix[word] = FloatTensor(np.random.uniform(0, 1, embedding_size))

    for i, w in enumerate(words_in_snli):
        if i % 2000 == 0:
            print('Creating embeddings: ' + str(i))
        if w in all_embeddings.keys():
            embeddings_matrix[w] = FloatTensor(all_embeddings[w])
        else:
            nos_of_unk_words += 1
            embeddings_matrix[w] = FloatTensor(embeddings_matrix['<unk>'])

    return embeddings_matrix, nos_of_unk_words, len(embeddings_matrix.keys())


def parse_dataset(file):
    glove = pandas.read_csv(file, sep=' ', header=None, quoting=csv.QUOTE_NONE,
                            lineterminator='\n')
    result_map = {}
    for index, row in glove.iterrows():
        row_as_list = row.tolist()
        result_map[row_as_list[0]] = row_as_list[1:201]
    return result_map
