import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, d_model],so need seq-len <= 5000
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self, input_dim=6, feature_size=100,num_layers=2,hidden_size=64,dropout=0.1, mean_dim=4, device="cpu"):
        super(TransAm, self).__init__()
        self._model_name = 'SimpleTransformer'
        self.input_embedding  = nn.Linear(input_dim,feature_size)
        self.src_mask = None
        self.meanDim = mean_dim
        self.covDim = int(mean_dim + mean_dim*(mean_dim - 1) / 2)
        self.dev=device

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,feature_size)
        self.init_weights()

        self.mean_output_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, mean_dim),
        )

        self.cov_output_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.covDim),
        )

        self.lower = torch.tensor([[float('-inf'), float('-inf'), -1., -1.]]).to(device)
        self.upper = torch.tensor([[float('inf'), float('inf'), 1., 1.]]).to(device)

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        # src with shape (input_window, batch_len, 1)
        if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
            self.src_mask = mask

        src = self.input_embedding(src) # linear transformation before positional embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        mean = self.mean_output_head(output)
        cov  = self.cov_output_head(output)
        mean =  torch.max(torch.min(mean, self.upper), self.lower)
        return mean, cov

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @property
    def model_name(self):
        return self._model_name

class skipTransAm(nn.Module):
    def __init__(self, input_dim=6, feature_size=30,num_layers=5,hidden_size=32,dropout=0.3, mean_dim=4, skip_size=3, dt=0.1, device="cpu"):
        super(SkipTransAm, self).__init__()
        self._model_name = 'SkipSimpleTransformer'
        self.dt=dt
        self.input_embedding  = nn.Linear(input_dim,feature_size)
        self.start_pos_embedding = nn.Linear(skip_size,feature_size)
        self.src_mask = None
        self.meanDim = mean_dim
        self.covDim = int(mean_dim + mean_dim*(mean_dim - 1) / 2)
        self.dev=device

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=5, dropout=dropout, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,self.meanDim + self.covDim)
        self.init_weights()

        #self.mean_output_head = nn.Sequential(
        #    nn.Linear(feature_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, mean_dim),
        #)

        #self.cov_output_head = nn.Sequential(
        #    nn.Linear(feature_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, self.covDim),
        #)

        self.lower = torch.tensor([[float('-inf'), float('-inf'), -1., -1.]]).to(device)
        self.upper = torch.tensor([[float('inf'), float('inf'), 1., 1.]]).to(device)

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src, trg):
        # src with shape (input_window, batch_len, 1)
        if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[1]+1).to(device)
            self.src_mask = mask

        init_states = self.get_init_states(src).to(self.dev)
        init_states = torch.unsqueeze(trg[:,0,:], 1)
        init_projection = self.start_pos_embedding(init_states)

        src = self.input_embedding(src) # linear transformation before positional embedding
        src = torch.concat([init_projection, src], dim=1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)[:,:-1,:]
        #mean = self.mean_output_head(output)
        #cov  = self.cov_output_head(output)
        mean = output[:,:,:self.meanDim]
        cov = output[:,:,-self.covDim:]
        mean =  torch.max(torch.min(mean, self.upper), self.lower)
        return mean, cov

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @property
    def model_name(self):
        return self._model_name


    # Get the displacement from the initial state to the ego point using proprioceptive info(velocity)
    def get_init_states(self, x):
        init_ego_views = x[:,0,:]
        pos = torch.zeros(x.shape[0], 3).to(self.dev)
        for t in range(x.shape[1]):
            v_t = x[:,t,-2:].to(self.dev)
            pos[:,0] += v_t[:,0] * self.dt * torch.cos(pos[:,2])
            pos[:,1] += v_t[:,0] * self.dt * torch.sin(pos[:,2])
            pos[:,2]  = torch.remainder((pos[:,2] + v_t[:,1] * self.dt), 2*math.pi) - math.pi

        return torch.unsqueeze(pos, 1).detach()
