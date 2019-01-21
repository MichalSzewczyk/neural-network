import csv

import pandas


class EmbeddingGenerator:
    @staticmethod
    def get_embedding(data_frame, file_name):
        glove = pandas.read_csv(file_name, sep=' ', header=None, quoting=csv.QUOTE_NONE,
                                lineterminator='\n')
        eos = pandas.np.random.uniform(low=-1.0, high=1.0, size=200)
        bos = pandas.np.random.uniform(low=-1.0, high=1.0, size=200)
        unk = pandas.np.random.uniform(low=-1.0, high=1.0, size=200)
        result_map = EmbeddingGenerator.get_as_map(glove)
        result_map['<eos>'] = eos
        result_map['<bos>'] = bos
        result_map['<unk>'] = unk
        EmbeddingGenerator.count_and_log_all_wrods(data_frame)

        print('Supported words: ' + str(result_map))
        data_frame['sentence1'].apply(lambda array: EmbeddingGenerator.replace_unknown(array))
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
    def count_and_log_all_wrods(data_frame):
        result_list = []
        for array in data_frame['sentence1']:
            for word in array:
                result_list.append(word)
        for array in data_frame['sentence2']:
            for word in array:
                result_list.append(word)
        print('Number of words: ' + str(len(result_list)))

    @staticmethod
    def replace_unknown(array):
        pass
