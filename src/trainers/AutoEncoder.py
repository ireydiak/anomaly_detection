import random

import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import trange

from . import BaseTrainer
from torch import Tensor, optim
import numpy as np
from typing import List, Tuple
from ..distributions import multivariate_normal_pdf, estimate_GMM_params


class MLADTrainer(BaseTrainer):

    def train_iter(self, sample: torch.Tensor):
        pass

    def __init__(self, train_set, **kwargs):
        super(MLADTrainer, self).__init__(**kwargs)
        # Unlike other methods, MLAD requires the train_set as a parameter because of the re-sampling and clustering
        # that happens at the beginning of every epoch iteration
        self.train_set = train_set.to(self.device)
        self.D = self.train_set.shape[1]
        self.K = kwargs.get('K', 4)

    def fit_clusters(self, X: Tensor) -> np.ndarray:
        # TODO: Question: train common_net prior or just a single pass?
        Z = self.model.common_pass(X)
        clusters = KMeans(n_clusters=self.K, random_state=0).fit(Z.detach().cpu().numpy())
        return clusters.labels_

    def create_batches(self, X_1: Tensor, X_2: Tensor, Z_1: Tensor, Z_2: Tensor, metric_input) -> List[Tuple]:
        N = self.batch_size
        # Number of batches
        n_batch = np.int(len(X_1) // N)
        # Handle the case where len(X_1) / N yields a remainder
        overflow = (len(X_1) % N) - 1
        # Prepare the indices which will be used to split X_1 and X_2 in mini batches
        indices = [(i * N, (i + 1) * N) for i in range(0, n_batch)]
        # Last batch will contain remainder
        if overflow > 0:
            indices[-1] = (indices[-1][0], indices[-1][1] + overflow)
        assert indices[-1][1] == len(X_1) - 1
        return [(
            (X_1[start:end, :], X_2[start:end, :]),
            (Z_1[start:end, :], Z_2[start:end, :]),
            metric_input[start:end, :]
        ) for start, end in indices]

    def split_siamese(self, X_1, X_2, labels):
        # TODO: By shuffling again, are we not mixing the different clusters together?
        idx_1 = random.sample(range(0, len(X_1)), len(X_1))
        idx_2 = random.sample(range(0, len(X_2)), len(X_2))
        input_x1 = X_1[idx_1, :]
        input_x2 = X_2[idx_1, :]
        x_label = labels[idx_1, :]
        input_z1 = input_x1[idx_2, :]
        input_z2 = input_x2[idx_2, :]
        z_label = labels[idx_2, :]
        metric_label = (torch.abs(x_label - z_label) == 0).float()
        return input_x1, input_x2, input_z1, input_z2, metric_label

    def create_samples(self, clusters) -> List[Tuple]:
        assert self.D
        input_x1 = torch.zeros(0, self.D).to(self.device)
        input_x2 = torch.zeros(0, self.D).to(self.device)
        x_label = torch.zeros(0, 1).to(self.device)

        for i in range(0, self.K):
            [coding_index] = np.where(clusters == i)
            input_x1 = torch.vstack((input_x1, self.train_set[coding_index, :])).to(self.device)
            np.random.shuffle(coding_index)
            # TODO: we run the risk of training the model on the same data
            input_x2 = torch.vstack((input_x2, self.train_set[coding_index, :])).to(self.device)
            k_labels = (torch.ones([len(coding_index), 1]) * i).to(self.device)
            x_label = torch.vstack((x_label, k_labels)).to(self.device)

        input_x1, input_x2, input_z1, input_z2, metric_label = self.split_siamese(
            input_x1, input_x2, x_label
        )

        return self.create_batches(input_x1, input_x2, input_z1, input_z2, metric_label)

    def train(self, dataset: DataLoader):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            print("Epoch {}/{}".format(epoch + 1, self.n_epochs))
            clusters = self.fit_clusters(self.train_set)
            batches = self.create_samples(clusters)
            with trange(len(batches)) as t:
                for i, tup in enumerate(batches):
                    optimizer.zero_grad()

                    X_tup, Z_tup, metric_label = tup[0], tup[1], tup[2]
                    X_1, X_2 = X_tup[0].to(self.device).float(), X_tup[1].to(self.device).float()
                    Z_1, Z_2 = Z_tup[0].to(self.device).float(), Z_tup[1].to(self.device).float()
                    metric_labels = metric_labels.to(self.device).float()

                    loss = self.forward(X_1, X_2, Z_1, Z_2, metric_label)
                    loss.backward()
                    optimizer.step()

                    train_loss = loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss))
                    t.update()
        print("Finished training")

    def compute_density(self, X: Tensor, phi: Tensor, mu: Tensor, Sigma: Tensor):
        density = 0.0
        # TODO: replace loops by matrix operations
        for k in range(0, self.K):
            density += multivariate_normal_pdf(X, phi[k], mu[k], Sigma[k, :, :])
        return density

    def compute_densities(self, X: Tensor, phi: Tensor, mu: Tensor, Sigma: Tensor) -> np.ndarray:
        self.model.eval()
        Z = self.model.common_pass(X)
        print(f'calculating GMM densities using \u03C6={phi.shape}, \u03BC={mu.shape}, \u03A3={Sigma.shape}')
        densities = np.zeros(len(Z))
        # TODO: replace loops by matrix operations
        for i in range(0, len(Z)):
            densities[i] = self.compute_density(Z[i], phi, mu, Sigma)
        return densities

    def score(self, sample: torch.Tensor):
        self.model.eval()
        # 1- estimate GMM parameters
        Z = self.model.common_pass(sample)
        _, gmm_z = self.model.gmm_net(Z)
        phi, mu, Sigma = estimate_GMM_params(Z, gmm_z)
        # 2- compute densities based on computed GMM parameters
        return self.compute_densities(sample, phi, mu, Sigma)

    def forward(self, X_1: Tensor, X_2: Tensor, Z_1: Tensor, Z_2: Tensor, metric_labels: Tensor):
        com_meta_tup, err_meta_tup, gmm_meta_tup, dot_met, ex_meta_tup, rec_meta_tup = self.model(X_1, X_2, Z_1, Z_2)
        return self.model.loss(
            com_meta_tup,
            gmm_meta_tup,
            dot_met,
            ex_meta_tup,
            rec_meta_tup,
            ((X_1, X_2), (Z_1, Z_2)),
            metric_labels
        )
