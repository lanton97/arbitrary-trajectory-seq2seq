import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import GRU, GELU
import math
from torch.nn.functional import relu
import torch.nn.functional as F
from roboticstoolbox.mobile import Unicycle
import numpy as np

# A simple GRU encoder for a Seq2SeqModel
class GRUEncoderModel(nn.Module):
    def __init__(self, 
                 inputSize,
                 hiddenSize=256,
                 num_layers=2,
                 dropout_p=0.1,
                 device="cpu",
                 ):
        super(GRUEncoderModel, self).__init__()
        # Validate input
        self.inputDim = inputSize
        self.hiddenSize = hiddenSize
        self.gru = GRU(inputSize, hiddenSize, batch_first=True, dropout=dropout_p, num_layers=num_layers)

    def forward(self, x):
        # Get the attention inputs
        x = x.type(torch.float)
        output, hidden = self.gru(x)
        return output, hidden

# A GRU Decoder for the Seq2Serq model
class GRUDecoderModel(nn.Module):
    def __init__(self, 
                 inputSize,
                 hiddenSize,
                 seqLen, # Length of the trajectories
                 # We have a default output dimension of 4 - sin(\th), cos(th) and x, y
                 # components of the homogenous transformation matrix
                 # I may try doing the whole matrix if this doesnt work with
                 # the losses
                 meanDim,
                 covDim,
                 device="cpu",
                 ):
        super(GRUDecoderModel, self).__init__()
        self.seqLen = seqLen
        self.meanDim = meanDim
        self.covDim = covDim
        self.hiddenSize=hiddenSize
        self.device=device
        self.inputSize = inputSize

        self.gru = GRU(inputSize, hiddenSize, batch_first=True)
        self.out = nn.Linear(hiddenSize, inputSize)
        self.mean_output_head = nn.Sequential(
            nn.Linear(inputSize, 128),
            nn.ReLU(),
            nn.Linear(128, meanDim),
        )

        self.cov_output_head = nn.Sequential(
            nn.Linear(inputSize, 128),
            nn.ReLU(),
            nn.Linear(128, self.covDim),
        )

    def forward(self, encoder_outputs, encoder_hidden):
        batch_size = encoder_outputs.size(0)
        # Our input is initially an empty(zeros) input
        decoder_input = torch.empty(batch_size, 1, self.inputSize, dtype=torch.long).to(self.device)
        decoder_hidden = torch.unsqueeze(encoder_hidden[0],0)
        decoder_outputs = []

        # Loop through each step in the sequence to get the decoder output
        for i in range(self.seqLen):
            # Get the GRU Output
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            # Detach to remove from gradient calculations
            decoder_input = decoder_output.detach()

        # Connect our outputs into a tensor
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # Use the GRU outputs with dense NNs to extract the mean and covariances
        means = self.mean_output_head(decoder_outputs)
        covs  = self.cov_output_head(decoder_outputs)
        return means, covs


    def forward_step(self, input, hidden):
        # Convert types and add nonlinearities
        input = relu(input.type(torch.float))
       
        # GRU Output
        output, hidden = self.gru(input, hidden)
        # Linear projection
        output = self.out(output)
        return output, hidden 

# A full Seq2Seq Model
class Seq2SeqModel(nn.Module):
    def __init__(self, 
                 inputDim,
                 encoder=GRUEncoderModel,
                 decoder=GRUDecoderModel,
                 numSteps=100,
                 hidden_size=64,
                 # We have a default output dimension of 4 - sin(\th), cos(th) and x, y
                 # components of the homogenous transformation matrix
                 # I may try doing the whole matrix if this doesnt work with
                 # the losses
                 meanDim=4,
                 dropout=0.0,
                 num_layer=1,
                 device="cpu",
                 skipSize=0
                 ):

        super(Seq2SeqModel, self).__init__()
        self._model_name = "seq2seq"
        # Validate input
        self.inputDim = inputDim
        self.seqLen = numSteps
        self.meanDim = meanDim
        self.covDim = int(meanDim + meanDim*(meanDim - 1) / 2)

        self.encoder = encoder(inputDim, num_layers=num_layer, hiddenSize=hidden_size, dropout_p=dropout, device=device)
        self.decoder = decoder(inputDim, hidden_size, self.seqLen, self.meanDim, self.covDim, device=device)

        # Used for clipping values to acceptable ranges for thetas
        self.lower = torch.tensor([[float('-inf'), float('-inf'), -1., -1.]]).to(device)
        self.upper = torch.tensor([[float('inf'), float('inf'), 1., 1.]]).to(device)

    def forward(self, x, init,):
        # Simple pass through the encoder and decoders
        out, hidden = self.encoder(x)
        means, covs = self.decoder(out, hidden)
        # Limit the thetas to a vald range
        means =  torch.max(torch.min(means, self.upper), self.lower)
        return means, covs

    @property
    def model_name(self):
        return self._model_name


# This GRU decoder uses the actual beginnning state as an input to the gru decoder 
# This is obtained in practice from a IEKF
class SkipToGRUDecoder(nn.Module):
    def __init__(self, 
                 inputSize,
                 hiddenSize,
                 seqLen,
                 # We have a default output dimension of 4 - sin(\th), cos(th) and x, y
                 # components of the homogenous transformation matrix
                 # I may try doing the whole matrix if this doesnt work with
                 # the losses
                 meanDim,
                 covDim,
                 skipSize=3, # The skip size refers to the shape of the 'location' input that is fed
                             # Directly into the GRU decoder
                 num_layers=2,
                 device="cpu",#.to(self.dev),
                 dropout=0.0,
                 ):
        super(SkipToGRUDecoder, self).__init__()
        self.seqLen = seqLen
        self.meanDim = meanDim
        self.covDim = covDim
        self.hiddenSize = hiddenSize
        self.device = device
        self._name = "skipGRUDecoder"

        self.gru = GRU(inputSize, hiddenSize, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.out = nn.Linear(hiddenSize, inputSize)
        self.initEmbed = nn.Linear(skipSize, inputSize)
        self.mean_output_head = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenSize, meanDim),
        )

        self.cov_output_head = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenSize, self.covDim),
        )

    def forward(self, encoder_outputs, encoder_hidden, initial_states):
        batch_size = encoder_outputs.size(0)
        # Our intitial state is passed into the decoder, allowing us to
        # Use a reasonable(noisy) estimate of the initial position
        decoder_input = self.initEmbed(initial_states.to(self.device))
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        # Loop through the encoder output for each timestep
        for i in range(self.seqLen):
            # Get our decoder outputs
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # Convert decoder outputs to means and covariances
        means = self.mean_output_head(decoder_outputs)
        covs  = self.cov_output_head(decoder_outputs)
        return means, covs


    def forward_step(self, input, hidden):
        input = input.type(torch.float)
        
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden 


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class skipSeq2SeqModel(nn.Module):
    def __init__(self, 
                 inputDim,
                 encoder=GRUEncoderModel,
                 decoder=SkipToGRUDecoder,
                 numSteps=100,
                 hidden_size=64,
                 # We have a default output dimension of 4 - sin(\th), cos(th) and x, y
                 # components of the homogenous transformation matrix
                 # I may try doing the whole matrix if this doesnt work with
                 # the losses
                 meanDim=4,
                 skipSize=4,
                 dt=0.0,
                 dropout=0.1,
                 num_layer=1,
                 device="cpu"
                 ):

        super(skipSeq2SeqModel, self).__init__()

        # Validate input
        self.inputDim = inputDim
        self.seqLen = numSteps
        self.meanDim = meanDim
        self.device = device
        self.dt = dt
        # Create encoder and decoder models
        self.covDim = int(meanDim + meanDim*(meanDim - 1) / 2)
        self.encoder = encoder(inputDim, num_layers=num_layer, hiddenSize=hidden_size, dropout_p=dropout, device=device)
        self.decoder = decoder(inputDim, hidden_size, self.seqLen, self.meanDim, self.covDim, device=device, dropout=dropout, num_layers=num_layer, skipSize=skipSize)
        self._model_name = "skipseq2seq" + "/"+self.decoder._name 

        self.lower = torch.tensor([[float('-inf'), float('-inf'), -1., -1.]]).to(device)
        self.upper = torch.tensor([[float('inf'), float('inf'), 1., 1.]]).to(device)

    def forward(self, x, trg, train=False):
        out, hidden = self.encoder(x)
        if train:
            trg += np.random.normal(0,0.2, trg.shape)
            trg = trg.float()
        init_states = torch.unsqueeze(trg[:,0,:], 1)
        means, covs = self.decoder(out, hidden, init_states)
        # Clip thetas to valid range
        means =  torch.max(torch.min(means, self.upper), self.lower)
        return means, covs

    @property
    def model_name(self):
        return self._model_name


