from torch import nn, FloatTensor
import torch


class LSTM_NLI(nn.Module):
    def __init__(self, word2embed, id2train, id2valid, batch_size, max_len_sent, embedding_dim, hidden_dim, output_dim):
        super(LSTM_NLI, self).__init__()
        self.word2embed = word2embed
        self.id2train = id2train
        self.id2valid = id2valid
        self.batch_size = batch_size
        self.max_len_sent = max_len_sent
        self.input_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.1)
        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden(self.hidden_dim, batch_size)

    def init_hidden(self, hidden_size, batch_size=1):
        return (torch.zeros(1, batch_size, hidden_size),
                torch.zeros(1, batch_size, hidden_size))

    def get_embeddings_for_batch(self, sentences, mode):
        print(sentences.shape)
        id2words = self.id2train if mode == 'train' else self.id2valid
        batch = FloatTensor()
        for i, sent in enumerate(sentences):
            for w in sent:
                batch = torch.cat((batch, self.word2embed[id2words[w.item()]]), dim=-1)
        return batch.reshape(self.batch_size, self.max_len_sent, -1)

    def forward(self, sentences, mode):
        word_embeds = self.get_embeddings_for_batch(sentences, mode)
        lstm_out, self.hidden = self.lstm(word_embeds, self.hidden)
        sent_embeds = self.hidden2out(self.hidden[0].contiguous().view((self.batch_size, -1)))
        print(sent_embeds.shape)
        return sent_embeds
