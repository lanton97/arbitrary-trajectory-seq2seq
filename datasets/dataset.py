import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets.dataset_utils import *
import numpy as np
from .construct_iekf_targets import constructIEKFTargets

class convoyDataset(Dataset):
    def __init__(self,
                 file_path="datasets/convoy.csv",
                 leader_speed=True,
                 iekf=False,
                 iekf_targets=False):
        df = load_dataset(file_path)
        df = select_features(df)
        df = splitIntoWindows(df)
        self.iekf_targets = iekf_targets
        if iekf:
            self.globalPos = getIEKFFeatures(df)
        if iekf_targets:
            items = getIEKFTargets(df)
            self.iekf_trgs = constructIEKFTargets(items)

        self.inputs, self.labels = constructInputTargetPairs(df, leader_speed=leader_speed)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.iekf_targets:
            return (self.inputs[idx], self.labels[idx], self.iekf_trgs[idx])
        return (self.inputs[idx], self.labels[idx])

    def getIEKFItems(self, idx):
        return self.globalPos[idx]

