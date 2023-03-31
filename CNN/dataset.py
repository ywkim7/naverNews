import numpy as np
import torch
from torch.utils.data import Dataset


class naverDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X_tensor = torch.stack(X, dim=0)
        self.Y_tensor = torch.stack(Y, dim=0)

    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index):
        return self.X_tensor[index], self.Y_tensor[index]
    

def naverCollator(dataset):
    input_list = []
    label_list = []
    
    for input, label in dataset:
        input_list.append(input)
        label_list.append(label)

    return torch.stack(input_list, dim=0), torch.stack(label_list, dim=0)