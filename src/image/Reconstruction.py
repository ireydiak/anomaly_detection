from src.image import BaseModel
import torch.nn as nn
import torch

from src.image.MemoryModule import MemoryModule


class MemAE(BaseModel):
    # input channel number
    def __init__(self, chnum_in: int, mem_dim: int, feature_dim: int):
        super(MemAE).__init__()
        self.memory, self.encoder, self.decoder = None, None, None
        self.chnum_in = chnum_in
        self.mem_dim = mem_dim
        self.feature_dim = feature_dim
        self.make_network()

    def make_network(self):
        n_features = 128
        n_features_2 = 96
        n_features_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, n_features_2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(n_features_2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(n_features_2, n_features, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(n_features),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(n_features, n_features_x2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(n_features_x2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(n_features_x2, n_features_x2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(n_features_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.memory = MemoryModule(mem_dim=self.mem_dim, feature_dim=n_features_x2, shrink_tresh=self.shrink_tresh)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(n_features_x2, n_features_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(n_features_x2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(n_features_x2, n_features, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(n_features),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(n_features, n_features_2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(n_features_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(n_features_2, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1))
        )

    def forward(self, X: torch.Tensor):
        z = self.encoder(X)
        res_mem = self.memory(z)
        z_hat, att = res_mem['output'], res_mem['att']
        x_hat = self.decoder(z_hat)
        return {'output': x_hat, 'att': att}
