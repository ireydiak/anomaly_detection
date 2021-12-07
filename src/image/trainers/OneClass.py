import logging

import torch
from torch.utils.data import DataLoader

from src.image.trainers import BaseTrainer


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, hypersphere_radius=None, hypersphere_center=None, **kwargs):
        super(DeepSVDDTrainer, self).__init__(**kwargs)
        self.R = torch.Tensor(hypersphere_radius, device=self.device)if hypersphere_radius is not None else None
        self.c = torch.Tensor(hypersphere_center, device=self.device) if hypersphere_center is not None else None

    def before_training(self, dataset: DataLoader):
        if self.c is None:
            logger = logging.getLogger()
            logger.info("initializing center c...")
            self.c = self.init_hypersphere_center(dataset)
            logger.info("initializing center c...")

    def train_iter(self, sample: torch.Tensor):
        logits = self.model(sample)
        dist = torch.sum((logits - self.c) ** 2, dim=1)
        return dist.mean()

    def score(self, sample: torch.Tensor):
        logits = self.model(sample)
        dist = torch.sum((logits - self.c) ** 2, dim=1)
        return dist

    def init_hypersphere_center(self, dataset: DataLoader, eps=0.1):
        N = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for row in dataset:
                X, _ = row
                X = X.to(self.device)
                logits = self.model(X)
                N += logits.shape[0]
                c += torch.sum(logits, dim=0)

        if c.isnan().sum() > 0:
            raise Exception("NaN value encountered during init_center_c")

        c /= N
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.model.train()

        return c

    def get_params(self) -> dict:
        return {'c': self.c, 'R': self.R, **self.model.get_params()}