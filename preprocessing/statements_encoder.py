import numpy as np
import torch


class StatementsEncoder:
    @staticmethod
    def encode_statements(data_frame):
        sentence_1_matrix_list = data_frame['sentence1'].as_matrix()
        sentence_2_matrix_list = data_frame['sentence2'].as_matrix()
        sentence_1_tensor_list = []
        for matrix in sentence_1_matrix_list:
            sentence_1_tensor_list.append(torch.tensor(matrix))
        sentence_2_tensor_list = []
        for matrix in sentence_2_matrix_list:
            sentence_2_tensor_list.append(torch.tensor(matrix))
        joined_matrix_list = []
        for sentence_1_tensor, sentence_2_tensor in zip(sentence_1_tensor_list, sentence_2_tensor_list):
            joined_matrix_list.append(sentence_1_tensor + sentence_2_tensor)
        print('FOO: ' + str(joined_matrix_list))
        # sentence_tensor_1 = torch.from_numpy(data_frame['sentence1'].as_matrix())
        # sentence_tensor_2 = torch.from_numpy(data_frame['sentence2'].as_matrix())
        # print('Tensor:')
        # print('Tensor1: ' + str(sentence_tensor_1))
        # print('Tensor2: ' + str(sentence_tensor_2))
        return data_frame

    @staticmethod
    def get_as_map(matrix):
        for words in matrix:
            for words2 in words:
                for word in words2:
                    print('Word: ' + str(type(word)))
        return matrix
