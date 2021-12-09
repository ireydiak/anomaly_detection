import os

import numpy as np
from torch.utils.data.dataset import T_co
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset, Subset


class IEEFraudDetection(Dataset):

    ad_label = "isFraud"
    index_col = "TransactionID"

    def __init__(self, root: str, train: bool, normal_class: int = None):
        self.root = root
        self.train = train
        self.normal_class = normal_class
        self.data, self.targets = self.load_data(root)
        self.anomaly_ratio = (self.targets == 1).sum() / len(self.targets)
        self.num_features = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index], self.targets[index]

    def load_data(self, path: str):
        if not path.endswith(".csv"):
            raise RuntimeError(f"Could not open {path}. This dataset can only read .csv files.")

        idx = None
        df = pd.read_csv(path, index_col=self.index_col)
        X = df.drop(columns=[self.ad_label], errors='ignore').values
        y = df.values if self.train else np.zeros(X.shape[0])

        if self.normal_class:
            idx = np.where(y == self.normal_class)

        return (X[idx], y[idx]) if idx is not None else (X, y)

    def shape(self):
        return self.data.shape

    def loaders(self,
                test_pct: float = 0.5,
                label: int = 0,
                batch_size: int = 128,
                num_workers: int = 0,
                seed: int = None) -> DataLoader:
        return DataLoader(dataset=self, batch_size=batch_size, num_workers=num_workers)
