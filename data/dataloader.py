from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super()

        self.df = df
        self.items = sorted(df.index.unique())
        self.attributes = sorted(df.columns.unique())

        self.NUM_ITEMS = len(self.items)
        self.NUM_ATTRIBUTES = len(self.attributes)

        self.features = torch.eye(self.NUM_ITEMS, dtype=torch.float32)
        self.targets = torch.tensor(df.values, dtype=torch.float32)

        assert len(self.targets) == len(self.features)

        self.n_samples = len(self.targets)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.features[index], self.targets[index])


def make_dataloader(df: pd.DataFrame) -> DataLoader:
    dataset = CustomDataset(df)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, collate_fn=lambda batch: batch)
    return dataloader
