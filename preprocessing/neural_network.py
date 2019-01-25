from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.target_size = target_size = 3
        self.dropout = nn.Dropout(0.5)
        self.layers = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            self.relu,
            self.dropout,
            nn.Linear(hidden_size, target_size),
            self.softmax
        )
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        return self.layers(inputs)
