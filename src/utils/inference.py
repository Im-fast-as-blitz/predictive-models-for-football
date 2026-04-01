import torch
import numpy as np

def test(model, loader, device, criterion):
    loss_log = []
    acc_log = []
    model.eval()

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
          out = model(data)
          loss = criterion(out, target)

          pred = out.argmax(dim=1)
          acc = (pred == target).float().mean()

          acc_log.append(acc.item())
          loss_log.append(loss.item())

    return np.mean(loss_log), np.mean(acc_log)
