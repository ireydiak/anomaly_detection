import neptune.new as neptune
from sklearn import metrics
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch
from torch.nn import Parameter
import numpy as np
from tqdm import trange
import torch.nn as nn
from src.trainers import BaseTrainer

#
# class DSEBMTrainer:
#
#     def __init__(self, D: int, model, lr: float = 1e-4, n_epochs: int = 100,
#               batch_size: int = 128, n_jobs_dataloader: int = 0, device: str = 'cuda'
#     ):
#         self.device = device
#         self.model = model.to(device)
#         self.batch_size = batch_size
#         self.n_jobs_dataloader = n_jobs_dataloader
#         self.n_epochs = n_epochs
#         self.lr = lr
#         self.b_prime = Parameter(batch_size, D)
#         self.optim = optim.Adam(
#             self.model.parameters() + [self.b_prime],
#             lr=lr, betas=(0.5, 0.999)
#         )
#
#     def train(self, dataset: DataLoader):
#         for _ in range(self.n_epochs):
#             with trange(len(dataset)) as t:
#                 for X, _ in dataset:
#                     X = X.to(self.device).float()
#                     noise = torch.randn(X.shape)
#                     X_noise = (torch.rand_like(X) + noise).to(self.device).float()
#                     net_out_noise = self.model(X_noise)
#                     energy_noise = 0.5 * torch.sum(torch.square(X - self.b_prime)) - net_out_noise
#                     grad_noise = torch.autograd.grad(energy_noise, X, retain_graph=True, create_graph=True)
#                     fx_noise = X_noise - grad_noise
#                     loss = torch.mean(torch.sum(torch.square(X - fx_noise)))
#
#                     self.optim.zero_grad()
#                     loss.backward()
#                     self.optim.step()
#                     t.set_postfix(loss='{:05.3f}'.format(loss))
#                     t.update()
#         print("Finished training")


class DSEBMTrainer(BaseTrainer):

    def __init__(self, D: int, score: str, **kwargs):
        super(DSEBMTrainer, self).__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss()
        self.b_prime = Parameter(torch.Tensor(self.batch_size, D).to(self.device))
        torch.nn.init.xavier_normal_(self.b_prime)
        self.score_name = score

    def set_optimizer(self):
        return optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train_iter(self, sample: torch.Tensor):
        noise = self.model.random_noise_like(sample).to(self.device)

        X_noise = sample + noise
        sample.requires_grad_()
        X_noise.requires_grad_()

        # out = self.model(sample)
        out_noise = self.model(X_noise)

        # energy = self.energy(sample, out)
        energy_noise = self.energy(X_noise, out_noise)

        dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
        fx_noise = (X_noise - dEn_dX[0])
        loss = self.loss(sample, fx_noise)
        return loss

    def loss(self, X, fx_noise):
        out = torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def score(self, sample: torch.Tensor):
        score_name = self.score_name.lower()[0]
        if score_name == 'e':
            # Energy-based score
            flat = sample - self.b_prime
            out = self.model(sample)
            score = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)
        else:
            # Reconstruction error
            sample.requires_grad_()
            out = self.model(sample)
            dEn_dX = torch.autograd.grad(self.energy(sample, out), sample)[0]
            score = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)

        return score

    def predict(self, scores: np.array, thresh: float):
        return (scores < thresh).astype(int)

    def energy(self, X, X_hat):
        return 0.5 * torch.sum(torch.square(X - self.b_prime)) - torch.sum(X_hat)
