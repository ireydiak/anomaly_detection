import numpy as np
from abc import abstractmethod
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import T_co
from typing import Tuple
from torch.utils.data import DataLoader


class AbstractDataset(Dataset):

    def __init__(self, path: str, pct: float = 1.0, **kwargs):
        X = self.load_data(path)
        if np.isnan(X).sum() > 0:
            print("Dropping {} rows with NaN values".format(np.isnan(X).sum()))
            X = X[~np.isnan(X).any(axis=1)]
        anomaly_label = kwargs.get('anomaly_label', 1)
        normal_label = kwargs.get('normal_label', 0)

        if pct < 1.0:
            # Keeps `pct` percent of the original data while preserving
            # the same normal/anomaly ratio
            anomaly_idx = np.where(X[:, -1] == anomaly_label)[0]
            normal_idx = np.where(X[:, -1] == normal_label)[0]
            np.random.shuffle(anomaly_idx)
            np.random.shuffle(normal_idx)

            X = np.concatenate(
                (X[anomaly_idx[:int(len(anomaly_idx) * pct)]],
                 X[normal_idx[:int(len(normal_idx) * pct)]])
            )
            self.X = X[:, :-1]
            self.y = X[:, -1]
        else:
            self.X = X[:, :-1]
            self.y = X[:, -1]

        self.anomaly_ratio = (X[:, -1] == anomaly_label).sum() / len(X)
        self.N = len(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index], self.y[index]

    def load_data(self, path: str):
        if path.endswith(".npz"):
            return np.load(path)[self.npz_key()]
        else:
            raise RuntimeError(f"Could not open {path}. This dataset can only read .npz files.")

    def D(self):
        return self.X.shape[1]

    def shape(self):
        return self.X.shape

    def get_data_index_by_label(self, label):
        return np.where(self.y == label)[0]

    def loaders(self,
                test_pct: float = 0.5,
                label: int = 0,
                batch_size: int = 128,
                num_workers: int = 0,
                seed: int = None) -> (DataLoader, DataLoader):
        train_set, test_set = self.split_train_test(test_pct, label, seed)
        train_ldr = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers)
        test_ldr = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers)
        return train_ldr, test_ldr

    def split_train_test(self, test_pct: float = .5, label: int = 0, seed=None) -> Tuple[Subset, Subset]:
        assert (label == 0 or label == 1)

        if seed:
            torch.manual_seed(seed)

        # Fetch and shuffle indices of a single class
        label_data_idx = np.where(self.y == label)[0]
        shuffled_idx = torch.randperm(len(label_data_idx)).long()

        # Generate training set
        num_test_sample = int(len(label_data_idx) * test_pct)
        num_train_sample = int(len(label_data_idx) * (1. - test_pct))
        train_set = Subset(self, label_data_idx[shuffled_idx[num_train_sample:]])

        # Generate test set based on the remaining data and the previously filtered out labels
        remaining_idx = np.concatenate([
            label_data_idx[shuffled_idx[:num_test_sample]],
            np.where(self.y == int(not label))[0]
        ])
        test_set = Subset(self, remaining_idx)

        return train_set, test_set

    @abstractmethod
    def npz_key(self):
        raise NotImplementedError
