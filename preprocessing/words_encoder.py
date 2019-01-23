import csv
import random

import pandas


class WordsEncoder:
    @staticmethod
    def prepare_embedding(file_name):
        glove = pandas.read_csv(file_name, sep=' ', header=None, quoting=csv.QUOTE_NONE,
                                lineterminator='\n')
        eos = [random.uniform(-1, 1) for _ in range(199)]
        bos = [random.uniform(-1, 1) for _ in range(199)]
        unk = [random.uniform(-1, 1) for _ in range(199)]
        result_map = WordsEncoder.get_as_map(glove)
        result_map['<eos>'] = eos
        result_map['<bos>'] = bos
        result_map['<unk>'] = unk
        # WordsEncoder.count_and_log_all_words(data_frame)
        # data_frame = WordsEncoder.replace_unknown_words(data_frame, result_map)
        # WordsEncoder.replace_words_with_vectors(data_frame, result_map)

        return result_map

    @staticmethod
    def get_as_map(glove):
        result_map = {}
        for index, row in glove.iterrows():
            row_as_list = row.tolist()
            result_map[row_as_list[0]] = row_as_list[1:200]
        return result_map

    @staticmethod
    def count_and_log_all_words(data_frame):
        result_list = []
        for array in data_frame['sentence1']:
            for word in array:
                result_list.append(word)
        for array in data_frame['sentence2']:
            for word in array:
                result_list.append(word)
        print('Number of words: ' + str(len(result_list)))

    @staticmethod
    def replace_words_with_vectors(data_frame, result_map):
        data_frame['sentence1'] = data_frame['sentence1'].apply(
            lambda x: WordsEncoder.replace_words_vector_with_matrix(x, result_map))
        data_frame['sentence2'] = data_frame['sentence2'].apply(
            lambda x: WordsEncoder.replace_words_vector_with_matrix(x, result_map))

    @staticmethod
    def replace_words_vector_with_matrix(words, result_map):
        result_list = []
        for word in words:
            result_list.append(result_map[word])
        return result_list


