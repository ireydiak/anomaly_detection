from torch import nn
from .BaseModel import BaseModel
from .MemoryModule import MemoryUnit
import torch
import numpy as np
import torch.nn.functional as F
from ...utils import weights_init_xavier


class MemAE(BaseModel):

    def __init__(self, in_features: int, mem_dim: int = 50, rep_dim: int = 1, device='cuda'):
        super(MemAE, self).__init__()
        self.in_features = in_features
        self.mem_dim = mem_dim
        self.device = device
        self.rep_dim = rep_dim
        self._build_net()

    def _build_net(self):
        D, L = self.in_features, self.rep_dim
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
            'in_features': self.in_features
        }


class DAGMM(BaseModel):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """

    def __init__(self, in_features: int, latent_dim: int, K: int, device='cuda'):
        """
        DAGMM constructor

        Parameters
        ----------
        in_features: int
            input dimension
        latent_dim: int
            latent dimension
        K: int
            number of mixtures
        device: str
            'cuda' or 'cpu'
        """
        super(DAGMM, self).__init__()
        self.latent_dim = latent_dim
        self.in_features = in_features
        # defaults to parameters described in section 4.3 of the paper
        # https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
        # if not ae_layers:
        #     enc_layers = [(input_size, 60, nn.Tanh()), (60, 30, nn.Tanh()), (30, 10, nn.Tanh()), (10, 1, None)]
        #     dec_layers = [(1, 10, nn.Tanh()), (10, 30, nn.Tanh()), (30, 60, nn.Tanh()), (60, input_size, None)]
        # else:
        #     enc_layers = ae_layers[0]
        #     dec_layers = ae_layers[1]
        #
        # gmm_layers = gmm_layers or [(3, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]
        #
        # self.ae = AE(enc_layers, dec_layers)
        # self.gmm = GMM.GMM(gmm_layers)

        self.cosim = nn.CosineSimilarity()
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.K = K
        self.device = device
        self._build_net()

    def print_name(self):
        return 'DAGMM'

    def _build_net(self):
        D, L, K = self.in_features, self.latent_dim, self.K
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
        self.gmm = nn.Sequential(
            nn.Linear(L + 2, 10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, self.K),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """

        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        z = self.enc(x)
        x_prime = self.dec(z)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([z, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        # compute gmm net output, that is
        #   - p = MLN(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm.forward(z_r)
        # gamma = self.softmax(output)

        return z, x_prime, cosim, z_r, gamma_hat

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def get_params(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "num_mixtures": self.K
        }


class NeuTraAD(BaseModel):
    def __init__(self, in_features: int, temperature: float, dataset: str, n_layers=3, device='cuda', **kwargs):
        super(NeuTraAD, self).__init__()
        self.device = device
        self.in_features = in_features
        self.latent_dim = in_features // 2
        self.n_layers = n_layers
        self.dataset = dataset
        self.K, self.Z, self.emb_out_dims = self._resolve_params(dataset)
        self.temperature = temperature
        self.cosim = nn.CosineSimilarity()
        self._build_network()
        self.enc.apply(weights_init_xavier)

    def print_name(self):
        return 'neuTraAD'

    def _create_masks(self) -> list:
        masks = [None] * self.K
        out_dims = np.array([self.in_features] * self.n_layers)
        for K_i in range(self.K):
            net_layers = create_network(self.in_features, out_dims, bias=False)
            net_layers[-1] = nn.Sigmoid()
            masks[K_i] = nn.Sequential(*net_layers).to(self.device)
        return masks

    def _build_network(self):
        # Encoder
        out_dims = self.emb_out_dims
        enc_layers = create_network(self.in_features, out_dims)[:-1]  # remove ReLU from the last layer
        self.enc = nn.Sequential(*enc_layers).to(self.device)
        # Masks / Transformations
        self.masks = self._create_masks()

    def _resolve_params(self, dataset: str) -> (int, int, list):
        K, Z = 7, 32  # num_transformers,
        out_dims = [90, 70, 50] + [Z]
        if dataset == 'Thyroid':
            Z = 12
            K = 4
            out_dims = [60, 40] + [Z]
        elif dataset == 'Arrhythmia':
            K = 11
            out_dims = [60, 40] + [Z]
        elif dataset == 'IEEEFraudDetection':
            K = 11
            out_dims = np.linspace(self.in_features, self.latent_dim, self.n_layers, dtype=np.int)
        return K, Z, out_dims

    def get_params(self) -> dict:
        return {
            'in_features': self.in_features,
            'num_transformations': self.K,
            'temperature': self.temperature
        }

    def score(self, X: torch.Tensor):
        Xk = self._computeX_k(X)
        Xk = Xk.permute((1, 0, 2))
        Zk = self.enc(Xk)
        Z = self.enc(X)
        Hij = self._computeBatchH_ij(Zk)
        Hx_xk = self._computeBatchH_x_xk(Z, Zk)

        mask_not_k = (~torch.eye(self.K, dtype=torch.bool, device=self.device)).float()
        numerator = Hx_xk
        denominator = Hx_xk + (mask_not_k * Hij).sum(dim=2)
        scores_V = numerator / denominator
        score_V = (-torch.log(scores_V)).sum(dim=1)

        return score_V

    def _computeH_ij(self, Z):
        hij = F.cosine_similarity(Z.unsqueeze(1), Z.unsqueeze(0), dim=2)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeBatchH_ij(self, Z):
        hij = F.cosine_similarity(Z.unsqueeze(2), Z.unsqueeze(1), dim=3)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeH_x_xk(self, z, zk):
        hij = F.cosine_similarity(z.unsqueeze(0), zk)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeBatchH_x_xk(self, z, zk):
        hij = F.cosine_similarity(z.unsqueeze(1), zk, dim=2)
        exp_hij = torch.exp(
            hij / self.temperature
        )
        return exp_hij

    def _computeX_k(self, X):
        X_t_s = []
        for k in range(self.K):
            X_t_k = self.masks[k](X) * X
            X_t_s.append(X_t_k)
        X_t_s = torch.stack(X_t_s, dim=0)

        return X_t_s

    def forward(self, X: torch.Tensor):
        return self.score(X)


def create_network(D: int, out_dims: np.array, bias=True) -> list:
    net_layers = []
    previous_dim = D
    for dim in out_dims:
        net_layers.append(nn.Linear(previous_dim, dim, bias=bias))
        net_layers.append(nn.ReLU())
        previous_dim = dim
    return net_layers
