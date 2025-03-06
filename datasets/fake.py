
import os
import torch
from torch.utils.data import Dataset
from datasets.dataset_utils import *
import numpy as np

class testDataset(Dataset):
    def __init__(self,
                 numData=10,
                 seqLen=100,
                 inputSize=5,
                 targetSize=3,
                 ):
        self.inputs = torch.zeros(numData, seqLen, inputSize)#)torch.rand(numData, seqLen, inputSize)* 10.0
        sequence = torch.arange(0, seqLen) * 0.1
        self.targets = (torch.ones(seqLen)*sequence)
        self.targets = torch.reshape(torch.repeat_interleave(self.targets, targetSize), (seqLen, targetSize))
        self.labels = self.targets.repeat(numData, 1, 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.labels[idx])
