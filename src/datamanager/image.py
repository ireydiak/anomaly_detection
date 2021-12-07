from abc import abstractmethod, ABC

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from PIL import Image


class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset
        self.anomaly_ratio = None

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader


class CIFAR10Manager:

    def __init__(self, normal_class: str = 'cat', download: bool = True, root: str = '../data'):
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        assert normal_class in classes
        self.N = None
        self.D = None
        self.download = download
        self.n_classes = 2
        self.normal_class = normal_class
        #self.normal_classes = (normal_class,)
        #self.outlier_classes = list(range(0, 10))
        #self.outlier_classes.remove(normal_class)
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

#
# class CIFAR10_inherited(CIFAR10):
#     """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""
#
#     def __init__(self, *args, **kwargs):
#         super(CIFAR10_inherited, self).__init__(*args, **kwargs)
#
#     def __getitem__(self, index):
#         """Override the original method of the CIFAR10 class.
#         Args:
#             index (int): Index
#         Returns:
#             triple: (image, target, index) where target is index of the target class.
#         """
#         if self.train:
#             img, target = self.train_data[index], self.train_labels[index]
#         else:
#             img, target = self.test_data[index], self.test_labels[index]
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, index  # only line changed
