"""
Author: Tu T. Do 
Email: tu.dothanh1906@gmail.com
"""

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# seeding
np.random.seed(1)

def _build_Xy_ionosphere():
    """
    Ionosphere dataset
    """
    df = pd.read_csv("datasets/ionosphere.data", header=None)
    X, y = df.iloc[:, 2:-1], df.iloc[:, -1]
    X, y = np.array(X), np.array(y)

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    return X, y, le


def _build_Xy_ecoli():
    """
    Ionosphere dataset
    """
    df = pd.read_csv("datasets/ecoli.data", header=None, sep=r"\s+")
    X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
    X, y = np.array(X), np.array(y)

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # Create imbalance dataset
    y = (y > 4) * 1

    return X, y, le


def _build_Xy_frog():
    """
    Ionosphere dataset
    """
    df = pd.read_csv("datasets/Frogs_MFCCs.csv")

    X = df.iloc[:, :22].values
    y = df.iloc[:, 22].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Convert to binary problem
    y = (y == 3) * 1

    return X, y, le


def _build_Xy_breast_tissue():
    """
    Ionosphere dataset
    """
    df = pd.read_excel("datasets/breast_tissue.xls", sheet_name=1)
    X, y = df.iloc[:, 3:].values, df.iloc[:, 1].values
    X, y = np.array(X), np.array(y)

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le


def _build_Xy_heart_2cl():
    """
    Ionosphere dataset
    """
    URL = "datasets/spectf.train"
    URL_ = "datasets/spectf.test"
    train = pd.read_csv(URL, header=None, sep=",")
    valid = pd.read_csv(URL_, header=None, sep=",")

    df = pd.concat([train, valid])
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X, y = np.array(X), np.array(y)

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, le


def _build_Xy_connectionist():
    df = pd.read_csv("datasets/sonar.all-data", header=None, sep=",")
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X, y = np.array(X), np.array(y)

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le


def _build_Xy_parkinsons():
    df = pd.read_csv("datasets/parkinsons.data", sep=",")
    df.drop(["name"], axis=1, inplace=True)
    X, y = df.drop(["status"], axis=1), df["status"]
    X, y = X.values, y.values

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le


def _build_Xy_balance():
    df = pd.read_csv("datasets/balance_3cl.csv", header=None, sep=",")
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X, y = np.array(X), np.array(y)

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le


def _build_Xy_breast_cancer():
    df = pd.read_csv("datasets/wdbc.data", header=None, sep=",")
    X, y = df.iloc[:, 2:].values, df.iloc[:, 1].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le


def _build_Xy(name: str, valsplit: float = 0.2) -> Dataset:
    """ """
    if name == "ionosphere":
        X, y, le = _build_Xy_ionosphere()

    elif name == "ecoli":
        X, y, le = _build_Xy_ecoli()

    elif name == "frogs":
        X, y, le = _build_Xy_frog()

    elif name == "breast-tissue":
        X, y, le = _build_Xy_breast_tissue()

    elif name == "heart_2cl":
        X, y, le = _build_Xy_heart_2cl()

    elif name == "connectionist":
        X, y, le = _build_Xy_connectionist()

    elif name == "parkinsons":
        X, y, le = _build_Xy_parkinsons()

    elif name == "balance":
        X, y, le = _build_Xy_balance()

    elif name == "breast-cancer":
        X, y, le = _build_Xy_breast_cancer()

    else:
        raise NotImplementedError(name)

    (X_train, X_valid, y_train, y_valid) = train_test_split(
        X, y, test_size=valsplit, random_state=1, stratify=y
    )

    return (X_train, y_train), (X_valid, y_valid), le


class DatasetFromNumpyArray(Dataset):
    """
    Dataset from numpy array (Feature, Label)
    """
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
    """
    Build Lightning DataModule
    """
    def __init__(self, name: str, valsplit: float, batch_size: int, num_workers: int):
        super().__init__()
        self.name = name
        self.valsplit = valsplit
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        (X_train, y_train), (X_valid, y_valid), le = _build_Xy(self.name, self.valsplit)
        self._train = X_train, y_train
        self._valid = X_valid, y_valid
        self.le = le  # label encoder

    def setup(self, stage: str):
        if stage == "fit":
            self.train = DatasetFromNumpyArray(*self._train)
            self.valid = DatasetFromNumpyArray(*self._valid)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid, batch_size=self.batch_size, num_workers=self.num_workers
        )


def build_datamodules(
    name: str, val_split: float, batch_size: int, num_workers: int
) -> L.LightningDataModule:
    """
    Interface to build datamodule from other function
    """

    return DataModuleFromNumpyArray(name, val_split, batch_size, num_workers)
