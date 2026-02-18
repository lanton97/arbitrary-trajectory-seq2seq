import torch
import torch.nn as nn
import math

class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim=6, hidden_size=64, num_layers=4, mean_dim=3, device="cpu", skipSize=None):
        super(SimpleRNNModel, self).__init__()
        self._model_name = 'SimpleRNN'
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.meanDim = mean_dim
        self.covDim = int(mean_dim + mean_dim*(mean_dim - 1) / 2)
        self.dev = device

        self.rnn = nn.RNN(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, mean_dim)
        self.init_weights()

        self.lower = torch.tensor([[float('-inf'), float('-inf'), -1.]]).to(device)
        self.upper = torch.tensor([[float('inf'), float('inf'), 1.]]).to(device)

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg=None, train=False):
        # src: (batch, seq_len, input_dim)
        output, _ = self.rnn(src)
        output = self.decoder(output)
        #mean = torch.max(torch.min(output, self.upper), self.lower)
        return output, None

    @property
    def model_name(self):
        return self._model_name
