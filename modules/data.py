import pandas as pd

import lightning as L
import torch
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder


# seeding
np.random.seed(1)

def _build_Xy(name: str, valsplit: float=.2) -> Dataset:
    """
    """
    if name == "ionosphere":
        df = pd.read_csv("datasets/ionosphere.data", header=None)
        X, y = df.iloc[:, 2:-1], df.iloc[:, -1]
        X, y = np.array(X), np.array(y)

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        # Number of samples
        N, _ = X.shape

        # Index of validation samples: 0 -> train, 1->valid
        valid_idx = np.random.choice(2, size=N, p=[1-valsplit, valsplit])
        X_train, y_train = X[valid_idx==0,:], y[valid_idx==0]
        X_valid, y_valid = X[valid_idx==1,:], y[valid_idx==1]

        return (X_train, y_train), (X_valid, y_valid), le

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
        y = torch.tensor(y, dtype=torch.int64)

        return x, y

    def __len__(self):
        N, _ = self.X.shape
        return N


class DataModuleFromNumpyArray(L.LightningDataModule):
    def __init__(
            self,
            name: str,
            valsplit: float,
            batch_size: int,
            num_workers: int):

        super().__init__()
        self.name = name
        self.valsplit = valsplit
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        (X_train, y_train), (X_valid, y_valid), le\
                = _build_Xy(self.name, self.valsplit)
        self.train = DatasetFromNumpyArray(X_train, y_train)
        self.valid = DatasetFromNumpyArray(X_valid, y_valid)
        self.le = le # label encoder


    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(
                self.train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
                self.valid,
                batch_size=self.batch_size,
                num_workers=self.num_workers)


def build_datamodules(
        name: str,
        val_split: float,
        batch_size: int,
        num_workers: int) -> L.LightningDataModule:

    return DataModuleFromNumpyArray(
            name, val_split, batch_size, num_workers)


if __name__ == "__main__":
    _build_Xy("ionosphere")
    # dm, le = build_datamodules("ionosphere", .2, 16, 4)
    # dm.prepare_data()
    # dm.setup("fit")

    # for (x, y) in dm.train_dataloader():
    #     print(x.shape, y.shape); break
