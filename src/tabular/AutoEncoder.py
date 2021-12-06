from torch import nn
from . import MemoryUnit
from . import BaseModel
import torch
import numpy as np
from pytorch_metric_learning import distances


class MemAE(BaseModel):

    def __init__(self, D: int, mem_dim: int = 50, rep_dim: int = 1, device='cuda'):
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


class DAGMM(BaseModel):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """

    def __init__(self, D: int, L: int, K: int, device='cuda'):
        """
        DAGMM constructor

        Parameters
        ----------
        D: int
            input dimension
        L: int
            latent dimension
        K: int
            number of mixtures
        device: str
            'cuda' or 'cpu'
        """
        super(DAGMM, self).__init__()
        self.L = L
        self.D = D
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
        D, L, K = self.D, self.L, self.K
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
            "L": self.L,
            "K": self.K
        }


class NeuTralAD(BaseModel):

    def __init__(self, D: int, N: int, dataset: str, device='cuda', n_layers: int = 3, temperature: float = 1.0):
        super(NeuTralAD, self).__init__()
        self.D = D
        self.N = N
        self.K, self.Z = self._resolve_params(dataset)
        self.temperature = temperature # TODO: try 0.07
        self.n_layers = n_layers
        self.device = device
        self.masks = []
        self.enc = None
        self._build_network()

    def __repr__(self):
        return 'NeuTralAD (D=%d, K=%d, Z=%d)' % (self.D, self.K, self.Z)

    def _build_network(self):
        # Encoder
        out_dims = np.linspace(self.D, self.Z, self.n_layers, dtype=np.int32)
        enc_layers = create_network(self.D, out_dims)
        self.enc = nn.Sequential(*enc_layers).to(self.device)
        # Masks / Transformations
        self.masks = self._create_masks()

    def _create_masks(self) -> list:
        masks = [0] * self.K
        out_dims = [self.D] * self.n_layers
        for K_i in range(self.K):
            net_layers = create_network(self.D, out_dims)
            net_layers[-1] = nn.Sigmoid()
            masks[K_i] = nn.Sequential(*net_layers).to(self.device)
        return masks

    def _resolve_params(self, dataset: str) -> (int, int):
        K, Z = 7, 32
        if dataset == 'Thyroid':
            Z = 12
            K = 4
        elif dataset == 'Arrhythmia':
            K = 11
        else:
            raise Exception('Unrecognized dataset %s' % dataset)
        return K, Z

    def score(self, X: torch.Tensor):
        # TODO: Bottleneck, replace with matrix operations
        total_sum = []
        for x in X:
            x = x.unsqueeze(0)
            sum_x = []
            for k in range(self.K):
                mask = self.masks[k]
                x_k = mask(x) * x
                numerator = (h_func(self.enc(x_k), self.enc(x))).squeeze(1) / self.temperature
                denominator = [(h_func(self.enc(x_k), self.enc(self.masks[j](x) * x)) / self.temperature).squeeze(1) for
                               j in range(self.K) if j != k]
                denominator = torch.Tensor(denominator).to(self.device)
                sum_x.append(
                    torch.log(numerator / (numerator + denominator.sum()))
                )
            total_sum.append(torch.stack(sum_x).sum())
        return -torch.stack(total_sum)

    def forward(self, X: torch.Tensor):
        return self.score(X).mean()

    def get_params(self) -> dict:
        return {
            'D': self.D,
            'N': self.N,
            'K': self.K,
            'temperature': self.temperature
        }


def create_network(D: int, out_dims: np.array) -> list:
    net_layers = []
    previous_dim = D
    for dim in out_dims:
        net_layers.append(nn.Linear(previous_dim, dim))
        net_layers.append(nn.ReLU())
        previous_dim = dim
    return net_layers


def h_func(x_k, x_l):
    dis = distances.CosineSimilarity()
    mat = dis(x_k, x_l)

    return torch.exp(
        mat
    )
