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
        mean = torch.max(torch.min(output, self.upper.to(output.device)), self.lower.to(output.device))
        return mean, None

    @property
    def model_name(self):
        return self._model_name


class BiDirRNNProbModel(BiDirRNNModel):
    def __init__(self, input_dim=6, hidden_size=128, num_layers=3, mean_dim=4, device="cpu", skipSize=None, dropout=0.15):
        super(BiDirRNNProbModel, self).__init__(input_dim, hidden_size=hidden_size, num_layers=num_layers,
                                                mean_dim=mean_dim, device=device, skipSize=skipSize, dropout=dropout)
        self.prob_output = True
        self._model_name = 'BiDirRNN_SE2'
        self.cov_decoder = nn.Linear(hidden_size * 2, self.covDim)
        self.cov_decoder.bias.data.zero_()
        self.cov_decoder.weight.data.uniform_(-0.1, 0.1)
        # Clip sin/cos to the valid range, as in the Seq2Seq model
        self.lower = torch.tensor([[float('-inf'), float('-inf'), -1., -1.]]).to(device)
        self.upper = torch.tensor([[float('inf'), float('inf'), 1., 1.]]).to(device)

    def forward(self, src, trg=None, train=False):
        src = src.float()
        output, _ = self.rnn(src)
        output = self.layer_norm(output)
        output = self.dropout(output)
        output = torch.relu(output)
        mean = self.decoder(output)
        mean = torch.max(torch.min(mean, self.upper.to(mean.device)), self.lower.to(mean.device))
        cov = self.cov_decoder(output)
        return mean, cov
