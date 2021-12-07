from src.image.models import BaseModel
import torch
import torch.nn as nn


class DeepSVDD(BaseModel):

    def __init__(self, rep_dim: int):
        super(DeepSVDD, self).__init__()
        self.conv_net = None
        self.mlp_net = None
        self.rep_dim = rep_dim
        self.build_network()

    def build_network(self):
        self.conv_net = nn.Sequential(
            # First layer
            nn.Conv2d(3, 32, (5, 5), bias=False, padding=2),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            # Second layer
            nn.Conv2d(32, 64, (5, 5), bias=False, padding=2),
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            # Third layer
            nn.Conv2d(64, self.rep_dim, (5, 5), bias=False, padding=2),
            nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.mlp_net = nn.Linear(self.rep_dim * 4 * 4, self.rep_dim, bias=False)

    def forward(self, X: torch.Tensor):
        x = self.conv_net(X)
        x = x.view(x.size(0), -1)
        return self.mlp_net(x)
