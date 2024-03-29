import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .LSTM import LSTM
from .NeuralNet import NeuralNet


class Model(nn.Module):
    def __init__(self, word2embed, id2train, id2valid, n_target, max_len_sent):
        super(Model, self).__init__()
        self.use_cuda = False
        self.word_embed_dim = 200
        self.hidden_dim_lstm = 300
        self.sent_embed_dim = 200
        self.sent_embed_nn_dim = 2 * self.sent_embed_dim
        self.hidden_dim_nn = 2 * self.hidden_dim_lstm
        self.target_size = n_target

        self.sent_encoder = LSTM(word2embed, id2train, id2valid, 32, max_len_sent, self.word_embed_dim,
                                 self.hidden_dim_lstm, self.sent_embed_dim)
        self.neural_net = NeuralNet(self.sent_embed_nn_dim, self.hidden_dim_nn, self.target_size)

    def forward(self, word_inp, mode):
        encoder_output_premise = self.sent_encoder(word_inp[0], mode)
        encoder_output_premise = F.dropout(encoder_output_premise, 0.1, self.training)
        print(encoder_output_premise.shape)
        encoder_output_hypo = self.sent_encoder(word_inp[1], mode)
        encoder_output_hypo = F.dropout(encoder_output_hypo, 0.1, self.training)
        print(encoder_output_hypo.shape)
        encoded_pair_of_sents = torch.cat((encoder_output_premise, encoder_output_hypo), 1)
        print("concatenated shape ", encoded_pair_of_sents.shape)
        encoded_pair_of_sents = Variable(encoded_pair_of_sents)
        scores = self.neural_net(encoded_pair_of_sents)
        return scores
