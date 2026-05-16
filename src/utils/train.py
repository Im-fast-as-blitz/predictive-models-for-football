from tqdm import tqdm

def train_epoch(model, optimizer, train_loader, device, criterion):
    loss_log = []
    acc_log = []
    model.train()

    for node_features, adj, hist, timestamps, full_history, x_cat, target in tqdm(train_loader):
      node_features = node_features.to(device)
      adj = adj.to(device)
      hist = hist.to(device)
      x_cat = x_cat.to(device)
      target = target.to(device)

      optimizer.zero_grad()
      out = model(node_features, x_cat)

      loss = criterion(out, target)
      loss.backward()
      optimizer.step()

      pred = out.argmax(dim=1)
      acc = (pred == target).float().mean()
      acc_log.append(acc.item())
      loss_log.append(loss.item())
    return loss_log, acc_log
