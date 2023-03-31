import os
import datetime
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel
from dataset import *
from model import naverCNN
from train import *

def setup():
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
     
    return rank, world_size, local_rank


def main(X, Y, embedding_model):
    rank, world_size, local_rank = setup()

    torch.cuda.set_device(local_rank)

    epochs = 5
    batch_size = 32

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, stratify=Y, test_size=0.2)

    train_dataset = naverDataset(X_train, Y_train)
    test_dataset = naverDataset(X_test, Y_test)

    train_sampler = DistributedSampler(X_train, num_replicas=world_size, rank=rank)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=naverCollator)
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=naverCollator)

    

    weights = torch.FloatTensor(embedding_model.wv.vectors).to(local_rank)
    n_filters = 100
    filter_sizes = [4, 6, 8]
    output_dim = 5
    dropout = 0.5

    model = naverCNN(weights=weights, n_filters=n_filters, filter_sizes=filter_sizes, output_dim=output_dim, dropout=dropout)
    model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        start_time = datetime.datetime.now()

        train_loss, train_acc = train_model(train_dataloader, model, local_rank, optimizer, criterion)
        eval_loss, eval_acc = eval_model(test_dataloader, model, criterion, local_rank)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time

        print('Epoch: {}/{} | Elapsed: {:}'.format(epoch+1, epochs, elapsed_time))
        print('Train Loss {:.4f} | Train Acc {:.3f}'.format(train_loss, train_acc))
        print('Validation Loss {:.4f} | Validation Acc {:.3f}'.format(eval_loss, eval_acc))        

    


if __name__=="__main__":
    mp.set_start_method('spawn')

    with open('/app/PoC/CNN/dataset', 'rb') as f:
        dataset = pickle.load(f)
    
    with open('/app/PoC/CNN/embedding_model', 'rb') as f:
        embedding_model = pickle.load(f)

    X = list(dataset.keys())
    Y = list(dataset.values())

    X = [torch.from_numpy(np.asarray(sentence)) for sentence in X]
    Y = [torch.from_numpy(np.asarray(label)) for label in Y]
    
    main(X, Y, embedding_model)