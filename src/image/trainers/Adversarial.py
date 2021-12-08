import torch
import torch.nn as nn
from torch.autograd import Variable
from src.image.trainers.BaseTrainer import BaseTrainer
from torch.utils.data.dataloader import DataLoader
from torch import optim
import neptune.new as neptune
from tqdm import trange


class ALADTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(ALADTrainer, self).__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim_ge = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optim_d = optim.Adam(
            list(self.model.D_xz.parameters()) + list(self.model.D_zz.parameters()) +
            list(self.model.D_xx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train_iter_dis(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Compute loss
        # Discriminators Losses
        loss_dxz = self.criterion(out_truexz, y_true) + self.criterion(out_fakexz, y_fake)
        loss_dzz = self.criterion(out_truezz, y_true) + self.criterion(out_fakezz, y_fake)
        loss_dxx = self.criterion(out_truexx, y_true) + self.criterion(out_fakexx, y_fake)

        return loss_dxz + loss_dzz + loss_dxx

    def train_iter_gen(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Generator losses
        loss_gexz = self.criterion(out_fakexz, y_true) + self.criterion(out_truexz, y_fake)
        loss_gezz = self.criterion(out_fakezz, y_true) + self.criterion(out_truezz, y_fake)
        loss_gexx = self.criterion(out_fakexx, y_true) + self.criterion(out_truexx, y_fake)
        cycle_consistency = loss_gexx + loss_gezz

        return loss_gexz + cycle_consistency

    def train(self, dataset: DataLoader, nep: neptune.Run = None):
        self.model.train()

        print('Started training')
        for epoch in range(self.n_epochs):
            ge_loss = 0.
            d_loss = 0.
            with trange(len(dataset)) as t:
                for sample in dataset:
                    # Cleaning gradients
                    self.optim_ge.zero_grad()
                    self.optim_d.zero_grad()

                    X, _ = sample
                    X = X.to(self.device).float()

                    if len(X) < self.batch_size:
                        break

                    d_loss += self.train_iter_dis(X)
                    ge_loss += self.train_iter_gen(X)

                    # Backward pass
                    d_loss.backward()
                    ge_loss.backward()

                    if nep:
                        nep["training/metrics/batch/ge_loss"] = ge_loss
                        nep["training/metrics/batch/d_loss"] = d_loss

                    # Backpropagation
                    d_loss.backward()

                    self.optim_d.step()
                    self.optim_ge.step()

                    t.set_postfix(
                        d_loss='{:05.3f}'.format(d_loss.item()),
                        ge_loss='{:05.3f}'.format(ge_loss.item())
                    )
                    t.update()

    def train_iter(self, sample: torch.Tensor):
        pass

    def score(self, sample: torch.Tensor):
        pass
