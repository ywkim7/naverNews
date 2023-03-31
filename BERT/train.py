import datetime
import torch

def train_model(dataloader, model, local_rank, optimizer, scheduler, loss_fn):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for step, batch in enumerate(dataloader):
        start_time = datetime.datetime.now()

        batch_ids, batch_masks, batch_labels = batch
        batch_ids = torch.stack([batch_id.to(local_rank) for batch_id in batch_ids], dim=0)
        batch_masks = torch.stack([batch_mask.to(local_rank) for batch_mask in batch_masks], dim=0)
        batch_labels = torch.stack([batch_label.to(local_rank) for batch_label in batch_labels], dim=0)
        optimizer.zero_grad()

        outputs = model(batch_ids, batch_masks)

        loss = loss_fn(outputs, batch_labels)

        pred = torch.argmax(outputs, dim=1)
        correct = pred.eq(batch_labels)
        acc = correct.sum() / len(correct)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        if (step % 300 == 0) & step != 0:
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                if local_rank==0:
                     print('Iteration {}/{} -> Train Loss: {:.4f}, Accuracy: {:.3f}, Elapsed: {:}'.format \
                      (step, len(dataloader), epoch_loss/step, epoch_acc, elapsed_time))
                epoch_loss = 0
                epoch_acc = 0

    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

  


def eval_model(dataloader, model, local_rank, loss_fn):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch_ids, batch_masks, batch_labels = batch
            batch_ids = torch.stack([batch_id.to(local_rank) for batch_id in batch_ids], dim=0)
            batch_masks = torch.stack([batch_mask.to(local_rank) for batch_mask in batch_masks], dim=0)
            batch_labels = torch.stack([batch_label.to(local_rank) for batch_label in batch_labels], dim=0)
            
            outputs = model(batch_ids, batch_masks)

            loss = loss_fn(outputs, batch_labels)

            pred = torch.argmax(outputs, dim=1)
            correct = pred.eq(batch_labels)
            acc = correct.sum() / len(correct)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)