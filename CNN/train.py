import torch

def train_model(dataloader, model, local_rank, optimizer, criterion):
    criterion = criterion.to(local_rank)

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in dataloader:
        batch_ids, batch_labels = batch
        batch_ids = torch.stack([batch_id.to(local_rank) for batch_id in batch_ids], dim=0)
        batch_labels = torch.stack([batch_label.to(local_rank) for batch_label in batch_labels], dim=0)
        optimizer.zero_grad()

        outputs = model(batch_ids)

        loss = criterion(outputs, batch_labels)

        pred = torch.argmax(outputs, dim=1)
        correct = pred.eq(batch_labels)
        acc = correct.sum() / len(correct)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

  


def eval_model(dataloader, model, criterion, local_rank):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch_ids, batch_labels = batch
            batch_ids = torch.stack([batch_id.to(local_rank) for batch_id in batch_ids], dim=0)
            batch_labels = torch.stack([batch_label.to(local_rank) for batch_label in batch_labels], dim=0)
            
            logits = model(batch_ids)

            loss = criterion(logits, batch_labels)

            pred = torch.argmax(logits, dim=1)
            correct = pred.eq(batch_labels)
            acc = correct.sum() / len(correct)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)