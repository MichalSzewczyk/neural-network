import json, torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from .LSTM import LSTM_NLI
from .NeuralNet import NeuralNet


class Model(nn.Module):
    def __init__(self, hyperparams, word2embed, id2train, id2valid, n_target, max_len_sent, use_cuda=False):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.hyperparams = hyperparams
        self.word_embed_dim = 200
        self.hidden_dim_lstm = 300
        self.sent_embed_dim = 200
        self.sent_embed_nn_dim = 2 * self.sent_embed_dim
        self.hidden_dim_nn = 2 * self.hidden_dim_lstm
        self.target_size = n_target

        self.sent_encoder = LSTM_NLI(word2embed, id2train, id2valid, 32, max_len_sent, self.word_embed_dim,
                                     self.hidden_dim_lstm, self.sent_embed_dim)
        self.neural_net = NeuralNet(self.sent_embed_nn_dim, self.hidden_dim_nn, self.target_size)

    def forward(self, word_inp, mode):
        encoder_output_premise = self.sent_encoder(word_inp[0], mode)
        encoder_output_premise = F.dropout(encoder_output_premise, self.hyperparams.dropout, self.training)
        print(encoder_output_premise.shape)
        encoder_output_hypo = self.sent_encoder(word_inp[1], mode)  # batch_size x sent_embed_dim?
        encoder_output_hypo = F.dropout(encoder_output_hypo, self.hyperparams.dropout, self.training)
        print(encoder_output_hypo.shape)
        encoded_pair_of_sents = torch.cat((encoder_output_premise, encoder_output_hypo), 1)  # albo dim=1...
        encoded_pair_of_sents = Variable(encoded_pair_of_sents)
        if self.use_cuda:
            encoded_pair_of_sents = encoded_pair_of_sents.cuda()

        scores = self.neural_net(encoded_pair_of_sents)
        return scores

    def save_model(self, path):
        torch.save(self.sent_encoder.state_dict(), os.path.join(path, 'sent_embedder.pkl'))
        torch.save(self.neural_net.state_dict(), os.path.join(path, 'neural_net.pkl'))

    def load_model(self, path):
        self.sent_encoder.load_state_dict(torch.load(os.path.join(path, 'sent_embedder.pkl')))
        self.neural_net.load_state_dict(torch.load(os.path.join(path, 'neural_net.pkl')))

    def is_saved_model(self, path):
        return os.path.getsize(os.path.join(path, 'sent_embedder.pkl')) != 0
