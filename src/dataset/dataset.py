import pandas as pd
import torch
from torch.utils.data import Dataset
from enum import Enum



class PandasDataset(Dataset):
    def __init__(self, args, kfold_step, train):
        dataset_name = args["name"]
        none_fill = args["none_fill"]

        self.df = pd.read_csv(f"data/{dataset_name}.csv")
        
        self.df = self.df[self.df[args["data"]["season_column"]] <= args["data"]["val_season"]]
        max_tour = self.df[self.df[args["data"]["season_column"]] == args["data"]["val_season"]][args["data"]["tour_column"]].max()
        if train == "train":
            # берем все предыдущие дни
            self.df = self.df[~((self.df[args["data"]["season_column"]] == args["data"]["val_season"]) & (self.df[args["data"]["tour_column"]] >= max_tour - kfold_step - 1))]
        elif train == "val":
            # берем предпоследний день для kfold конкретный
            self.df = self.df[(self.df[args["data"]["season_column"]] == args["data"]["val_season"]) & (self.df[args["data"]["tour_column"]] == max_tour - kfold_step - 1)]
        elif train == "test":
            # берем последний день для kfold конкретный
            self.df = self.df[(self.df[args["data"]["season_column"]] == args["data"]["val_season"]) & (self.df[args["data"]["tour_column"]] == max_tour - kfold_step)]
        else:
            raise Exception("Unknown Split data strategy")
        
        self.df = self.df.drop(columns=args["data"]["igonre_columns"])

        self.target_col = args["data"]["target_column"]
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]
        
        if none_fill == "zero":
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        else:
            raise Exception("Unkonw strategy for fill none")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        features = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        target = torch.tensor(row[self.target_col], dtype=torch.long)

        return features, target