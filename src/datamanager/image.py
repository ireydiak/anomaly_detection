import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CIFAR10():
    def loaders(self,
                test_pct: float = 0.5,
                label: int = 0,
                batch_size: int = 4,
                num_workers: int = 0,
                seed: int = None) -> (DataLoader, DataLoader):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform
        )
        train_ldr = DataLoader(trainset, batch_size=batch_size)

        testset = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform
        )
        test_ldr = DataLoader(testset, batch_size=batch_size, shuffle=False)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_ldr, test_ldr
