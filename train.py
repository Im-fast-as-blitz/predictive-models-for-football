import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.dataset import PandasDataset
from src.utils import parse_args, get_optimizer, get_scheduler, get_criterion
from src.model import ITransformer
from src.utils import train_epoch, test

def train():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    kfold_steps = args["train"]["kfold_steps"]
    batch_size = args["train"]["batch_size"]
    n_epochs = args["train"]["n_epochs"]

    for i in range(kfold_steps):
        train_dataset = PandasDataset(args=args["dataset"], kfold_step=i+1, kfold_steps=kfold_steps, train="train")
        val_dataset = PandasDataset(args=args["dataset"], kfold_step=i+1, kfold_steps=kfold_steps, train="val")
        test_dataset = PandasDataset(args=args["dataset"], kfold_step=i+1, kfold_steps=kfold_steps, train="test")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = ITransformer(args["model"], len(train_dataset.feature_cols))
        optimizer = get_optimizer(args["optimizer"], model)

        if "scheduler" in args.keys():
            scheduler = get_scheduler(args["scheduler"], optimizer)
        else:
            scheduler = None

        criterion = get_criterion(args["criterion"])

        train_loss_log = []
        train_acc_log = []
        val_loss_log = []
        val_acc_log = []

        print("="*25)
        print(f"Fold number {i + 1}\n")

        for epoch in range(n_epochs):
            train_loss, train_acc = train_epoch(model, optimizer, train_dataloader, device, criterion)
            val_loss, val_acc = test(model, val_dataloader, device, criterion)

            train_loss_log.extend(train_loss)
            train_acc_log.extend(train_acc)

            val_loss_log.append(val_loss)
            val_acc_log.append(val_acc)

            print(f"Epoch {epoch}")
            print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
            print(f" val loss: {val_loss}, val acc: {val_acc}\n")

            if scheduler is not None:
                scheduler.step()

        test_loss, test_acc = test(model, test_dataloader, device, criterion)
        print("- " * 25)
        print(f"test loss: {np.mean(test_loss)}, test acc: {np.mean(test_acc)}")
        print("- " * 25)


if __name__ == "__main__":
    train()
