import torch
import torch.nn as nn
import math

class BiDirRNNModel(nn.Module):
    def __init__(self, input_dim=6, hidden_size=128, num_layers=3, mean_dim=3, device="cpu", skipSize=None, dropout=0.15):
        super(BiDirRNNModel, self).__init__()
        self.prob_output = False
        self._model_name = 'BiDirRNN'
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.meanDim = mean_dim
        self.covDim = int(mean_dim + mean_dim*(mean_dim - 1) / 2)
        self.dev = device

        self.rnn = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size * 2, mean_dim)
        self.init_weights()

        self.lower = torch.tensor([[float('-inf'), float('-inf'), -math.pi]]).to(device)
        self.upper = torch.tensor([[float('inf'), float('inf'), math.pi]]).to(device)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg=None, train=False):
        src = src.float()
        output, _ = self.rnn(src)
        output = self.layer_norm(output)
        output = self.dropout(output)
        output = torch.relu(output)
        output = self.decoder(output)
        mean = torch.max(torch.min(output, self.upper), self.lower)
        return mean, None

    @property
    def model_name(self):
        return self._model_name
