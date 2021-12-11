from abc import abstractmethod, ABC

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from PIL import Image


class CIFAR10Manager:

    def __init__(self, normal_class: str = 'cat', download: bool = True, root: str = '../data'):
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        assert normal_class in classes
        self.N = None
        self.D = None
        self.download = download
        self.n_classes = 2
        self.normal_class = normal_class
        self.root = root

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])

        train_set = CIFAR10(root=self.root, train=True, download=True, transform=transform)
        class_idx = train_set.class_to_idx['cat']

        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.where(train_set.targets == class_idx, 0, 1)
        train_idx_normal = np.argwhere(np.isin(train_set.targets, 0)).flatten().tolist()
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = CIFAR10(root=self.root, train=False, download=True, transform=transform)
        self.test_set.targets = np.array(self.test_set.targets)
        self.test_set.targets = np.where(self.test_set.targets == class_idx, 0, 1)

        self.shape = train_set.data.shape
        self.anomaly_ratio = 6000 / 60_000

    def loaders(self, batch_size: int = 4, num_workers: int = 0, seed: int = None) -> (DataLoader, DataLoader):
        train_ldr = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        test_ldr = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

        return train_ldr, test_ldr

