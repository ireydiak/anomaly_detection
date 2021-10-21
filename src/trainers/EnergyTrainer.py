from sklearn import metrics
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch
from torch.nn import Parameter
import numpy as np
from tqdm import trange

class DSEBMTrainer:
    
    def __init__(self, D: int, model, lr: float = 1e-4, n_epochs: int = 100,
              batch_size: int = 128, n_jobs_dataloader: int = 0, device: str = 'cuda'
    ):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.b_prime = Parameter(batch_size, D)
        self.optim = optim.Adam(
            self.model.parameters() + [self.b_prime],
            lr=lr, betas=(0.5, 0.999)
        )

    def train(self, dataset: DataLoader):
        for _ in range(self.n_epochs):
            with trange(len(dataset)) as t:
                for X, _ in dataset:
                    X = X.to(self.device).float()
                    noise = torch.randn(X.shape)
                    X_noise = (torch.rand_like(X) + noise).to(self.device).float()
                    net_out_noise = self.model(X_noise)
                    energy_noise = 0.5 * torch.sum(torch.square(X - self.b_prime)) - net_out_noise
                    grad_noise = torch.autograd.grad(energy_noise, X, retain_graph=True, create_graph=True)
                    fx_noise = X_noise - grad_noise
                    loss = torch.mean(torch.sum(torch.square(X - fx_noise)))

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    t.set_postfix(loss='{:05.3f}'.format(loss))
                    t.update()
        print("Finished training")
