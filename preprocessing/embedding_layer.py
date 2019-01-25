import numpy as np
import torch
from torch import nn


class EmbeddingLayerCreator:
    @staticmethod
    def create_layer(prepared_for_embedding, words_to_vectors_map):
        matrix_len = len(prepared_for_embedding)
        weights_matrix = np.zeros((matrix_len, 200))
        for i, word in enumerate(words_to_vectors_map):
            weights_matrix[i] = torch.tensor(words_to_vectors_map[word])
        weights_matrix_tensor = torch.tensor(weights_matrix)
        emb_layer = nn.Embedding(matrix_len, 200)
        emb_layer.load_state_dict({'weight': weights_matrix_tensor})
        emb_layer.weight.requires_grad = False
        return emb_layer
