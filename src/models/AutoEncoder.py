from torch import nn
from . import MemoryUnit
from . import BaseModel
import torch


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
