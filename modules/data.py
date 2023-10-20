import pandas as pd

import lightning as L
import torch
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import LabelEncoder


def _build_Xy(name: str) -> Dataset:
    """
    """
    if name == "ionosphere":
        df = pd.read_csv("datasets/ionosphere.data", header=None)
        X, y = df.iloc[:, 2:-1], df.iloc[:, -1]
        X, y = np.array(X), np.array(y)

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        return X, y, le

    else:
        raise NotImplementedError(name)


class DatasetFromNumpyArray(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X, self.y = X, y

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.to_list()

        x, y = self.X[idx, :], self.y[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int32)

        return x, y

    def __len__(self):
        N, _ = self.X.shape
        return N


class DataModuleFromNumpyArray(L.LightningDataModule):
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            val_split: float,
            batch_size: int,
            num_workers: int):

        super().__init__()
        self.full = DatasetFromNumpyArray(X, y)
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self): pass

    def setup(self, stage: str):
        if stage == "fit":
            N = len(self.full)
            Nv = int(self.val_split * N) 
            self.train, self.valid = random_split(
                    self.full, [N - Nv, Nv])

        if stage == "test":
            raise NotImplementedError()

        if stage == "predict":
            raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
                self.train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

    def valid_dataloader(self):
        return DataLoader(
                self.valid,
                batch_size=self.batch_size,
                num_workers=self.num_workers)


def build_datamodules(
        name: str,
        val_split: float,
        batch_size: int,
        num_workers: int) -> L.LightningDataModule:
    X, y, le = _build_Xy(name)
    return DataModuleFromNumpyArray(
            X, y, val_split, batch_size, num_workers), le




if __name__ == "__main__":
    dm, le = build_datamodules("ionosphere", .2, 16, 4)
    dm.prepare_data()
    dm.setup("fit")

    for (x, y) in dm.train_dataloader():
        print(x.shape, y.shape); break





