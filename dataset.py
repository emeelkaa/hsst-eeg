import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, root: str, files: list):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sample = pickle.load(open(os.path.join(self.root, self.files[idx]), 'rb'))
        X = sample['X']
        Y = sample['y']
        X = torch.FloatTensor(X)
        return X, Y
    

def get_chbmit():
    root = "chbmit/clean_segments_2"
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    train_dataset = EEGDataset(os.path.join(root, "train"), train_files)
    val_dataset = EEGDataset(os.path.join(root, "val"), val_files)
    test_dataset = EEGDataset(os.path.join(root, "test"), test_files)

    return train_dataset, val_dataset, test_dataset


def get_tuev():
    root = "tuev"
    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    train_dataset = EEGDataset(os.path.join(root, "processed_train"), train_files)
    val_dataset = EEGDataset(os.path.join(root, "processed_train"), val_files)
    test_dataset = EEGDataset(os.path.join(root, "processed_eval"), test_files)
    return train_dataset, val_dataset, test_dataset