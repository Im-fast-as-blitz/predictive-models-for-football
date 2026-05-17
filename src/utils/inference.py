import torch
import numpy as np

def test(model, loader, device, criterion):
    loss_log = []
    acc_log = []
    model.eval()

    for node_features, adj, hist, timestamps, full_history, x_cat, target in loader:
        node_features = node_features.to(device)
        adj = adj.to(device)
        hist = hist.to(device)
        x_cat = x_cat.to(device)
        timestamps = timestamps.to(device)
        full_history = full_history.to(device)

        target = target.to(device)

        with torch.no_grad():
            out = model(node_features, adj, hist, timestamps, full_history, x_cat)
            loss = criterion(out, target)

            pred = out.argmax(dim=1)
            acc = (pred == target).float().mean()

            acc_log.append(acc.item())
            loss_log.append(loss.item())

    return np.mean(loss_log), np.mean(acc_log)
