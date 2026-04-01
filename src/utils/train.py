def train_epoch(model, optimizer, train_loader, device, criterion):
    loss_log = []
    acc_log = []
    model.train()

    for data, target in train_loader:
      data = data.to(device)
      target = target.to(device)

      optimizer.zero_grad()
      out = model(data)

      loss = criterion(out, target)
      loss.backward()
      optimizer.step()

      pred = out.argmax(dim=1)
      acc = (pred == target).float().mean()
      acc_log.append(acc.item())
      loss_log.append(loss.item())
    return loss_log, acc_log
