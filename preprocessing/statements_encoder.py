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

        return data_frame
