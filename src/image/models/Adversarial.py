from torch.autograd import Variable

from src.image.models import BaseModel
import torch
import torch.nn as nn

from src.image.utils import weights_init_xavier


def get_padding(size, kernel_size, stride, dilation=1):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class ALAD(BaseModel):

    def __init__(self, feature_dim: int, latent_dim: int, *kwargs):
        super(ALAD, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.D_xz = DiscriminatorXZ(feature_dim, 128, latent_dim, negative_slope=0.2, p=0.5)
        self.D_xx = DiscriminatorXX(feature_dim, 128, negative_slope=0.2, p=0.2)
        self.D_zz = DiscriminatorZZ(latent_dim, latent_dim, negative_slope=0.2, p=0.2)
        self.G = Generator(latent_dim, feature_dim, negative_slope=1e-4)
        self.E = Encoder(feature_dim, latent_dim)
        self.D_xz.apply(weights_init_xavier)
        self.D_xx.apply(weights_init_xavier)
        self.D_zz.apply(weights_init_xavier)
        self.G.apply(weights_init_xavier)
        self.E.apply(weights_init_xavier)

    def forward(self, X: torch.Tensor):
        # Encoder
        z_gen = self.E(X)

        # Generator
        z_real = Variable(
            torch.randn((X.size(0), self.L)).to(self.device),
            requires_grad=False
        )
        x_gen = self.G(z_real)

        # DiscriminatorXZ
        out_truexz, _ = self.D_xz(X, z_gen)
        out_fakexz, _ = self.D_xz(x_gen, z_real)

        # DiscriminatorZZ
        out_truezz, _ = self.D_zz(z_real, z_real)
        out_fakezz, _ = self.D_zz(z_real, self.E(self.G(z_real)))

        # DiscriminatorXX
        out_truexx, _ = self.D_xx(X, X)
        out_fakexx, _ = self.D_xx(X, self.G(self.E(X)))

        return out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx


class Encoder(nn.Module):
    def __init__(self, in_features, latent_dim, negative_slope=0.2):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.negative_slope = negative_slope
        self.net = self.build_network()

    def build_network(self):
        return nn.Sequential(
            # First Layer
            nn.Conv2d(self.in_features, 128, kernel_size=(4, 4), padding=get_padding(128, 4, 2, 1), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.negative_slope),
            # Second Layer
            nn.Conv2d(128, 256, kernel_size=(4, 4), padding=get_padding(256, 4, 2, 1), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.negative_slope),
            # Third Layer
            nn.Conv2d(256, 512, kernel_size=(4, 4), padding=get_padding(512, 4, 2, 1), stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.negative_slope),
            # Fourth Layer
            nn.Conv2d(512, self.latent_dim, kernel_size=(4, 4), padding='valid', stride=1),
        )

    def forward(self, X: torch.Tensor):
        return self.net(X)


class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, negative_slope=1e-4):
        super(Generator, self).__init__()
        self.negative_slope = negative_slope
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.net = self.build_network()

    def build_network(self):
        return nn.Sequential(
            # First Layer
            nn.ConvTranspose2d(self.latent_dim, 512, kernel_size=(4, 4), stride=(2, 2), padding='valid'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Second Layer
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=get_padding(256, 4, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Third Layer
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=get_padding(128, 4, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Fourth Layer
            nn.ConvTranspose2d(128, self.feature_dim, kernel_size=(4, 4), stride=(2, 2), padding=get_padding(128, 4, 2)),
            nn.Tanh()
        )

    def forward(self, Z: torch.Tensor):
        return self.net(Z)


class DiscriminatorXZ(nn.Module):
    """
    Discriminates between pairs (E(x), x) and (z, G(z))
    """

    def __init__(self, in_features, out_features, latent_dim, negative_slope=0.2, p=0.5, n_classes=1):
        super(DiscriminatorXZ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.p = p
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.net_x, self.net_z, self.net_xz_1, net_xz_2 = None, None, None, None
        self.build_network()

    def build_network(self):
        self.net_x = nn.Sequential(
            nn.Conv2d(self.in_features, 128, kernel_size=(4, 4), stride=(2, 2), padding=get_padding(128, 4, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.negative_slope),

            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=get_padding(256, 4, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.negative_slope),

            nn.Conv2d(128, 512, kernel_size=(4, 4), stride=(2, 2), padding=get_padding(512, 4, 2)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.negative_slope)
        )
        self.net_z = nn.Sequential(
            # First Layer
            nn.Conv2d(self.latent_dim, 512, kernel_size=1, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # Second Layer
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.net_xz_1 = nn.Sequential(
            nn.Conv2d(2 * 512, 1024, kernel_size=1, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2)
        )
        self.net_xz_2 = nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding='same')

    def forward_xz(self, xz):
        mid_layer = self.net_xz_1(xz)
        logits = self.net_xz_2(mid_layer)
        return logits, mid_layer

    def forward(self, X: torch.Tensor, Z: torch.Tensor):
        # Inference over X
        x = self.net_x(X)
        # Inference over Z
        z = self.net_z(Z)
        # Joint inference
        xz = torch.cat((x, z), dim=1)
        return self.forward_xz(xz)


class DiscriminatorXX(nn.Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, p=0.5, n_classes=1):
        super(DiscriminatorXX, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.negative_slope = negative_slope
        self.p = p
        self.net_1, self.fc_1 = None, None
        self.build_network()

    def build_network(self):
        self.net_1 = nn.Sequential(
            # First Layer
            nn.Conv2d(2 * self.in_features, 64, kernel_size=5, stride=2, padding=get_padding(64, 5, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # Second Layer
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=get_padding(128, 5, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.fc_1 = nn.Linear(128, 1)

    def forward(self, X: torch.Tensor, rec_X: torch.Tensor):
        XX = torch.cat((X, rec_X), dim=1)
        mid_layer = self.net_1(XX)
        return self.fc_1(mid_layer), mid_layer


class DiscriminatorZZ(nn.Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, p=0.5, n_classes=1):
        super(DiscriminatorZZ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.negative_slope = negative_slope
        self.p = p
        self.n_classes = n_classes
        self.fc_1, self.fc_2 = None, None
        self.build_network()

    def build_network(self):
        self.fc_1 = nn.Sequential(
            nn.Linear(2 * self.in_features, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fc_1 = nn.Sequential(
            nn.Linear(32, 1),
        )

    def forward(self, Z, rec_Z):
        ZZ = torch.cat((Z, rec_Z), dim=1)
        mid_layer = self.fc_1(ZZ)
        return self.fc_2(mid_layer), mid_layer
