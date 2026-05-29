from tqdm import tqdm
import torch

def train_epoch(model, optimizer, train_loader, device, criterion, scheduler, logger, epoch):
    loss_log = []
    acc_log = []
    model.train()

    i = 0
    for node_features, adj, hist, _, full_history, x_cat, target in tqdm(train_loader):
      logger.set_step(epoch * len(train_loader) + i, mode="train")

      node_features = node_features.to(device)
      adj = adj.to(device)
      hist = hist.to(device)
      full_history = full_history.to(device)
      x_cat = x_cat.to(device)

      target = target.to(device)

      optimizer.zero_grad()
      out = model(node_features, adj, hist, full_history, x_cat)

      loss = criterion(out, target)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      pred = out.argmax(dim=1)
      acc = (pred == target).float().mean()
      acc_log.append(acc.item())
      loss_log.append(loss.item())

      logger.add_scalars({
          "train_loss": loss_log[-1],
          "train_accuracy": acc_log[-1],
          "lr": optimizer.param_groups[0]["lr"]
      })

      i += 1
      if scheduler is not None:
        scheduler.step()

    return loss_log, acc_log
