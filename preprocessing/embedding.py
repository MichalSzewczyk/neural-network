import csv
import random

import pandas


class EmbeddingGenerator:
    @staticmethod
    def get_embedding(data_frame, file_name):
        glove = pandas.read_csv(file_name, sep=' ', header=None, quoting=csv.QUOTE_NONE,
                                lineterminator='\n')
        eos = [random.uniform(-1, 1) for _ in range(201)]
        bos = [random.uniform(-1, 1) for _ in range(201)]
        unk = [random.uniform(-1, 1) for _ in range(201)]
        result_map = EmbeddingGenerator.get_as_map(glove)
        result_map['<eos>'] = eos
        result_map['<bos>'] = bos
        result_map['<unk>'] = unk
        EmbeddingGenerator.count_and_log_all_words(data_frame)
        data_frame = EmbeddingGenerator.replace_unknown_words(data_frame, result_map)
        EmbeddingGenerator.count_and_log_unknown_words(data_frame)

        print('Embedding data: ' + str(data_frame))
        return data_frame

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
    def count_and_log_unknown_words(data_frame):
        result_list = []
        for array in data_frame['sentence1']:
            for word in array:
                if word is '<unk>':
                    result_list.append(word)

        for array in data_frame['sentence2']:
            for word in array:
                if word is '<unk>':
                    result_list.append(word)

        print('Number of unknown words: ' + str(len(result_list)))

    @staticmethod
    def replace_unknown_words(data_frame, result_map):
        result_list = []
        data_frame['sentence1'] = data_frame['sentence1'].apply(
            lambda row: EmbeddingGenerator.replace_unknown(row, result_map, result_list))
        data_frame['sentence2'] = data_frame['sentence2'].apply(
            lambda row: EmbeddingGenerator.replace_unknown(row, result_map, result_list))
        print('FOO: Number of words: ' + str(len(result_list)))
        return data_frame

    @staticmethod
    def replace_unknown(array, result_map, replaced_words):
        result_array = []
        for word in array:
            if word is '<bos>' or word is '<eos>' or word in result_map.keys():
                result_array.append(word)
            else:
                result_array.append('<unk>')
                replaced_words.append(word)
        return result_array
