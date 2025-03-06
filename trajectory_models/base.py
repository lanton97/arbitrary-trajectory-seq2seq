from common.se2 import *
from abc import ABC
from abc import abstractmethod
import numpy as np
import torch

# A base interface for trajectory models
# Essentially just makes sure the inputs to the NN and outputs
# are the correct shape and type
class manifoldTrajectoryModel():
    def __init__(self,
                 model=None,
                 meanDim=4,
                 ):
        self.model=model
        self.meanDim = meanDim

    # This calls the internal NN model to return our input
    def getModelOutput(self, inp, trg):
        inp = torch.Tensor(inp)
        inp = torch.unsqueeze(inp, 0)
        trg = torch.Tensor(trg)
        trg = torch.unsqueeze(trg, 0)
        pred_means, pred_cov = self.model(inp,trg)
        # Reshape output into means and covariance matrices
        cov = self.constructCovariances(pred_cov)
        q_hat = torch.squeeze(pred_means)
        cov = torch.squeeze(cov)
        return q_hat, cov 

    # Calculate the covariance matrices from the pearon coefficients
    def constructCovariances(self, vec):
        cov = torch.zeros((vec.shape[0], vec.shape[1], self.meanDim, self.meanDim))#, requires_grad=True)
        for i in range(self.meanDim):
            auto_cov = torch.exp(vec[:,:,i])
            cov[:,:,i,i] = torch.exp(vec[:,:,i])

        # Keep track of our relevant pearson coeeficient
        pearsonCount = 0
        for i in range(self.meanDim):
            for j in range(i+1):
                # Skip if we already have the element from the diagonal covariance
                if i != j:
                    # Calculate the off-diagonal covariance matrix entry
                    cov[:,:,i,j] = torch.tanh(vec[:,:,self.meanDim + pearsonCount]) *torch.sqrt(torch.exp(vec[:,:,i])*torch.exp(vec[:,:,j]))
                    cov[:,:,j,i] = torch.tanh(vec[:,:,self.meanDim + pearsonCount]) *torch.sqrt(torch.exp(vec[:,:,i])*torch.exp(vec[:,:,j]))

                    pearsonCount += 1

        return cov

    def loadModel(self, filePath, name=''):
        self.model = torch.load(filePath + name + 'model.pt')

