import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import resample

class CHBMITDataset(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        X = sample["X"]
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        Y = torch.tensor(Y, dtype=torch.long)
        return X, Y

class TUEVDataset(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 250
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)+ 1e-8)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        Y = torch.tensor(Y, dtype=torch.long)
        return X, Y


