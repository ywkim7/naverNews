import os
import datetime
import pickle
import numpy as np
import torch
import torch.distributed as dist
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DistributedSampler, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel
from dataset import naverDataset, naverCollator
from model import naverModel
from train2 import train_model, eval_model

def setup():
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
     
    return rank, world_size, local_rank


def main(ids, masks, labels):
    rank, world_size, local_rank = setup()

    torch.cuda.set_device(local_rank)

    epochs = 5

    dataset = naverDataset(ids, masks, labels)
    train_size = int(len(dataset) * 0.8)
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = random_split(dataset, lengths=[train_size, eval_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, sampler=train_sampler, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, pin_memory=True)


    total_steps = len(train_dataloader) * epochs

    model = naverModel(num_classes=5, dr_rate=2e-5)
    model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        start_time = datetime.datetime.now()
        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_model(train_dataloader, model, local_rank, optimizer, scheduler, loss_fn)
        eval_loss, eval_acc = eval_model(eval_dataloader, model, local_rank, loss_fn)

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time

        if local_rank == 0:
            print('Epoch: {}/{} | Elapsed: {:}'.format(epoch+1, epochs, elapsed_time))
            print('Train Loss {:.4f} | Train Acc {:.3f}'.format(train_loss, train_acc))
            print('Validation Loss {:.4f} | Validation Acc {:.3f}'.format(eval_loss, eval_acc))    

    

    


if __name__=="__main__":
    with open('/app/PoC/BERT/input_label', 'rb') as f:
        dataset = pickle.load(f)

    inputs = list(dataset.keys())
    labels = list(dataset.values())

    ids = [torch.from_numpy(np.asarray(sentence[0])) for sentence in inputs]
    masks = [torch.from_numpy(np.asarray(sentence[1])) for sentence in inputs]
    labels = [torch.from_numpy(np.asarray(label)) for label in labels]
    
    main(ids, masks, labels)