from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, embed_size, hidden_size, target_size):
        super(NeuralNet, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.target_size = target_size = 3
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=1)
        self.layers = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            self.relu,
            self.dropout,
            nn.Linear(hidden_size, target_size),
            self.softmax
        )

    def forward(self, inputs):
        return self.layers(inputs)