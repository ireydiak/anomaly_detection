import logging
import time

from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch
import numpy as np


class DeepSVDDTrainer:

    def __init__(self, model, R, c,
                 lr: float = 1e-4, n_epochs: int = 100, batch_size: int = 128, n_jobs_dataloader: int = 0,
                 device: str = 'cuda'):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.c = c
        self.R = R

    def train(self, dataset):
        logger = logging.getLogger()
        train_ldr, _ = dataset.loaders(batch_size=self.batch_size, num_worker=self.n_jobs_dataloader)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_ldr)
            logger.info('Center c initialized.')

        logger.info('Started training')
        epoch_loss = 0.0
        epoch_start_time = time.time()
        for epoch in self.n_epochs:
            logger.info('')
            for x_i in train_ldr:
                inputs, _, _ = x_i
                inputs = inputs.to(self.device).float()

                # Reset gradient
                optimizer.zero_grad()

                outputs = self.model(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                loss = torch.mean(dist)

                # Backpropagation
                loss.backward()
                optimizer.step()

                epoch_loss += loss

            epoch_train_time = time.time() - epoch_start_time
            logger.info(
                'Epoch {}/{} \tTime: {:.3f}\t Loss: {:.8f}'.format(
                    epoch + 1, self.n_epochs, epoch_train_time, epoch_loss
                )
            )
        logger.info('Finished training')

    def test(self, dataset):
        pass

    def init_center_c(self, train_loader: DataLoader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
           Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for x_i in train_loader:
                # get the inputs of the batch
                inputs, _, _ = x_i
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
