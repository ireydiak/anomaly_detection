from torch import nn
from torch.nn.modules.activation import Tanh
from . import MemoryUnit
from . import BaseModel
import torch
from torch import Tensor
from .utils import create_network
from typing import Tuple


class AutoEncoder(BaseModel):
    def __init__(self, enc_layers, dec_layers):
        super(AutoEncoder, self).__init__()
        self.encoder = create_network(enc_layers)
        self.decoder = create_network(dec_layers)
        self.L = dec_layers[0][0]
        self.code_shape = enc_layers[-1][1]

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)

    def forward(self, X):
        z = self.encoder(X)
        x_prime = self.decoder(z)
        return x_prime, z


class MemAE(BaseModel):

    def __init__(self, D: int, mem_dim: int = 50, rep_dim: int = 1, device='cpu'):
        super(MemAE, self).__init__()
        self.D = D
        self.mem_dim = mem_dim
        self.device = device
        self.rep_dim = rep_dim
        self._build_net()

    def _build_net(self):
        D, L = self.D, self.rep_dim
        self.enc = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, D // 4),
            nn.Tanh(),
            nn.Linear(D // 4, L)
        ).to(self.device)
        self.dec = nn.Sequential(
            nn.Linear(L, D // 4),
            nn.Tanh(),
            nn.Linear(D // 4, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, D)
        ).to(self.device)
        self.mem_module = MemoryUnit(self.mem_dim, self.rep_dim)

    def forward(self, X: torch.Tensor):
        z = self.enc(X)
        z_hat, att = self.mem_module(z)
        x_hat = self.dec(z_hat)
        return x_hat, att

    def print_name(self) -> str:
        return "MemAE"

    def get_params(self) -> dict:
        return {
            'rep_dim': self.rep_dim,
            'D': self.D
        }


class MLAD(BaseModel):

    def __init__(
            self, D: int, L: int, K: int, **kwargs
    ):
        """

        Parameters
        ----------
        D: Number of features (n_features)
        L: Size of latent space (dimensionality of latent space)
        K: Number of gaussian mixtures
        """
        super(MLAD, self).__init__(**kwargs)
        # Common network
        self.common_net = create_network([
            (D, 64, nn.ReLU()),
            (64, 64, nn.ReLU()),
            (64, L, nn.Sigmoid())
        ])
        # self.common_net = nn.Sequential(
        #     nn.Linear(D, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, L),
        #     nn.Sigmoid()
        # )
        # Error network
        self.error_net = create_network([
            (D, 64, nn.ReLU()),
            (64, 64, nn.ReLU()),
            (64, L, nn.Sigmoid())
        ])
        # self.error_net = nn.Sequential(
        #     nn.Linear(D, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, L),
        #     nn.Sigmoid()
        # )
        # Representation network (Decoder)
        # repr_net_layers = kwargs.get(
        #     'representation_layers', [(2 * L, 64, nn.ReLU()), (64, 96, nn.ReLU()), (96, D, nn.Sigmoid())]
        # )
        self.reconstructor_net = create_network([
            (2 * L, 64, nn.ReLU()),
            (64, 96, nn.ReLU()),
            (96, D, nn.Sigmoid())
        ])
        # self.reconstructor = nn.Sequential(
        #     nn.Linear(2 * L, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 96),
        #     nn.ReLU(),
        #     nn.Linear(96, D, nn.Sigmoid())
        # )
        # Exchanger network (Decoder)
        # exchange_net_layers = kwargs.get(
        #     'exchanger_layers', [(L, 64, nn.ReLU()), (64, 64, nn.ReLU()), (64, D, nn.Sigmoid())]
        # )
        self.exchange_net = create_network([
            (L, 64, nn.ReLU()),
            (64, 64, nn.ReLU()),
            (64, D, nn.Sigmoid())
        ])
        # GMM network
        gmm_net_layers = kwargs.get('gmm_layers', [
            ((L, 16, nn.ReLU()), (16, 16, nn.ReLU()), (16, K, nn.Softmax(dim=1))),
            ((K, 16, nn.ReLU()), (16, 16, nn.ReLU()), (16, L, nn.Sigmoid()))
        ])
        self.gmm_net = AutoEncoder(gmm_net_layers[0], gmm_net_layers[1])
        self.lamb_1 = kwargs.get('lambda_1', 1e-04)
        self.lamb_2 = kwargs.get('lambda_2', 1)
        self.lamb_3 = kwargs.get('lambda_3', 1)
        self.lamb_4 = kwargs.get('lambda_4', 1e-04)
        self.lamb_5 = kwargs.get('lambda_5', 1e-04)

        self.K = K
        self.L = L
        self.D = D

    def get_params(self) -> dict:
        return {
            'latent_dim': self.L,
            'input_dim': self.D,
            'num_mixtures': self.K
        }

    def reconstruction_loss(self, X_1_hat: Tensor, X_2_hat: Tensor, X_1: Tensor, X_2: Tensor):
        loss = nn.MSELoss()
        return loss(X_1_hat, X_2) + loss(X_2_hat, X_1)

    def exchanger_loss(self, X_1_hat: Tensor, X_2_hat: Tensor, X_1: Tensor, X_2: Tensor):
        loss = nn.MSELoss()
        return loss(X_1, X_1_hat) + loss(X_2, X_2_hat)

    def gmm_loss(self, common_1: Tensor, common_2: Tensor, gmm_1: Tensor, gmm_2: Tensor):
        loss = nn.MSELoss()
        return self.lamb_4 * loss(common_1, gmm_1) + self.lamb_4 * loss(common_2, gmm_2)

    def common_loss(self, common_1: Tensor, common_2: Tensor):
        loss = nn.MSELoss()
        return loss(common_1, common_2)

    def metric_loss(self, dot_metric, metric_input):
        loss = nn.MSELoss()
        return loss(dot_metric, metric_input)

    def loss(self, com_meta_tup, gmm_meta_tup, dot_metrics, ex_meta_tup, rec_meta_tup, samples_meta_tup, metric_label):
        """
        Computes the loss from the siamese two-input streams

        Parameters
        ----------
        com_meta_tup: A Tuple of two-input Tensors (common)
        gmm_meta_tup: A Tuple of two-input Tensors (gmm)
        dot_metrics: A (n_sample, 1) Tensor
        ex_meta_tup: A Tuple of two-input Tensors (exchanger)
        rec_meta_tup: A Tuple of two-input Tensors (reconstructor)
        samples_meta_tup: A Tuple of two-input Tensors containing the original samples
        metric_label: A (n_sample, 1) Tensor

        Returns
        -------
        The sum of MSE loss for the five sub-networks
        """
        # Common Loss
        common_loss_A = self.common_loss(*com_meta_tup[0])
        common_loss_B = self.common_loss(*com_meta_tup[1])
        # Reconstruction Loss
        rec_loss_A = self.reconstruction_loss(*rec_meta_tup[0], *samples_meta_tup[0])
        rec_loss_B = self.reconstruction_loss(*rec_meta_tup[1], *samples_meta_tup[1])
        # Exchanger Loss
        ex_loss_A = self.exchanger_loss(*ex_meta_tup[0], *samples_meta_tup[0])
        ex_loss_B = self.exchanger_loss(*ex_meta_tup[1], *samples_meta_tup[1])
        # GMM Loss
        gmm_loss_A = self.gmm_loss(*com_meta_tup[0], *gmm_meta_tup[0])
        gmm_loss_B = self.gmm_loss(*com_meta_tup[1], *gmm_meta_tup[1])
        # Metric Loss
        metric_loss = sum([self.metric_loss(dot_metric, metric_label) for dot_metric in dot_metrics])
        # Compute losses
        loss_A = self.lamb_1 * common_loss_A + self.lamb_2 * rec_loss_A + self.lamb_3 * ex_loss_A + gmm_loss_A
        loss_B = self.lamb_1 * common_loss_B + self.lamb_2 * rec_loss_B + self.lamb_3 * ex_loss_B + gmm_loss_B
        loss_metric = self.lamb_5 * metric_loss
        return loss_A + loss_B + loss_metric

    def forward_one(self, X: Tensor) -> (float, float, float, float, float):
        """
        Performs a forward pass on a single input

        Parameters
        ----------
        X: A (n_inputs, n_features) Tensor

        Returns
        -------
        The output of the five MLAD networks.
        """
        # Common
        common = self.common_net(X)
        # Error
        err = self.error_net(X)
        # GMM, GMM coding
        gmm, gmm_z = self.gmm_net(common)
        # Exchanger
        ex = self.exchange_net(common)

        return common, err, gmm, gmm_z, ex

    def forward_two(self, X_1: Tensor, X_2: Tensor):
        """
        Performs a forward pass on a two-input stream

        Parameters
        ----------
        X_1: A (n_inputs, n_features) Tensor
        X_2: A (n_inputs, n_features) Tensor

        Returns
        -------
        Tuples from the five sub-networks
        """
        common_1, err_1, gmm_1, gmm_z_1, ex_1 = self.forward_one(X_1)
        common_2, err_2, gmm_2, gmm_z_2, ex_2 = self.forward_one(X_2)
        # Concat
        mix_1 = torch.cat((common_1, err_2), dim=1)
        mix_2 = torch.cat((common_2, err_1), dim=1)
        # Decode
        rec_1 = self.reconstructor_net(mix_1)
        rec_2 = self.reconstructor_net(mix_2)
        return (common_1, common_2), (err_1, err_2), (gmm_1, gmm_2), (gmm_z_1, gmm_z_2), (ex_1, ex_2), (rec_1, rec_2)

    def forward(self, X_1: Tensor, X_2: Tensor, Z_1: Tensor, Z_2: Tensor) -> Tuple[Tuple, Tuple, Tuple, Tuple, Tuple, Tuple]:
        """
        Performs a forward pass on the model.
        Since we are using a siamese two-input stream network, the forward pass is decoupled into multiple
        smaller functions.

        Parameters
        ----------
        X_1: A (n_inputs, n_features) Tensor
        X_2: A (n_inputs, n_features) Tensor
        Z_2: A (n_inputs, n_features) Tensor
        Z_1: A (n_inputs, n_features) Tensor

        Returns
        -------

        """
        common_tup_1, err_tup_1, gmm_tup_1, gmm_tup_z_1, ex_tup_1, rec_tup_1 = self.forward_two(X_1, X_2)
        common_tup_2, err_tup_2, gmm_tup_2, gmm_tup_z_2, ex_tup_2, rec_tup_2 = self.forward_two(Z_1, Z_2)
        dot_metrics = (
            (gmm_tup_z_1[0] * gmm_tup_z_2[0]) @ torch.ones(self.K, 1).to(self.device),
            (gmm_tup_z_1[0] * gmm_tup_z_2[1]) @ torch.ones(self.K, 1).to(self.device),
            (gmm_tup_z_1[1] * gmm_tup_z_2[0]) @ torch.ones(self.K, 1).to(self.device),
            (gmm_tup_z_1[1] * gmm_tup_z_2[1]) @ torch.ones(self.K, 1).to(self.device)
        )

        return (common_tup_1, common_tup_2), \
               (err_tup_1, err_tup_2), \
               (gmm_tup_1, gmm_tup_2), \
               dot_metrics, \
               (ex_tup_1, ex_tup_2), \
               (rec_tup_1, rec_tup_2)

    def common_pass(self, X):
        return self.common_net(X)
