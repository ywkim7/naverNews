import torch
from torch.utils.data import Dataset


class naverDataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids = torch.stack(ids, dim=0)
        self.masks = torch.stack(masks, dim=0)
        self.labels = torch.stack(labels, dim=0)


    def __len__(self):
        return self.ids.size(dim=0)
    

    def __getitem__(self, index):
        return self.ids[index], self.masks[index], self.labels[index]
    


def naverCollator(dataset):
    ids_list = []
    masks_list = []
    label_list = []

    for id, mask, label in dataset:
        ids_list.append(id)
        masks_list.append(mask)
        label_list.append(label)

    return torch.stack(ids_list, dim=0), torch.stack(masks_list, dim=0), torch.stack(label_list, dim=0)