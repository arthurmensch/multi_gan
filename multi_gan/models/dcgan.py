from torch import nn as nn

from gamesrl.utils import weights_init


class GeneratorDCGAN64(nn.Module):
    def __init__(self, in_features, out_channels, n_filters, batch_norm=True):
        super(GeneratorDCGAN64, self).__init__()

        if batch_norm:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(in_features, n_filters * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(n_filters * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(n_filters, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        else:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(in_features, n_filters * 8, 4, 1, 0, bias=False),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 4, 2, 1, bias=False),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(n_filters, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


class GeneratorDCGAN32(nn.Module):
    def __init__(self, in_features, out_channels, n_filters, batch_norm=True):
        super(GeneratorDCGAN32, self).__init__()

        if batch_norm:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(in_features, n_filters * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(n_filters * 4),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 2),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(n_filters, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 32 x 32
            )
        else:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(in_features, n_filters * 4, 4, 1, 0, bias=False),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 4, 2, 1, bias=False),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(n_filters, out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 32 x 32
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

    def forward(self, input):
        return (self.main(input) + 1) / 2


class DiscriminatorDCGAN64(nn.Module):
    def __init__(self, in_channels, n_filters, batch_norm=True):
        super(DiscriminatorDCGAN64, self).__init__()
        if batch_norm:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(in_channels, n_filters, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(n_filters * 8, 1, 4, 1, 0, bias=False),
            )
        else:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(in_channels, n_filters, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(n_filters * 8, 1, 4, 1, 0, bias=False),
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class DiscriminatorDCGAN32(nn.Module):
    def __init__(self, in_channels, n_filters, n_targets, batch_norm=True):
        super(DiscriminatorDCGAN32, self).__init__()
        if batch_norm:
            self.main = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(in_channels, n_filters, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 16 x 16
                nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 8 x 8
                nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 4
                nn.Conv2d(n_filters * 4, n_targets, 4, 1, 0, bias=False),
            )
        else:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(in_channels, n_filters, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 16 x 16
                nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 8 x 8
                nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 4
                nn.Conv2d(n_filters * 4, 1, 4, 1, 0, bias=False),
            )
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class DiscriminatorDCGAN28(nn.Module):
    def __init__(self, in_channels, n_filters, n_targets):
        super(DiscriminatorDCGAN28, self).__init__()
        self.n_targets = n_targets
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(in_channels, n_filters, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(n_filters, n_filters * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(n_filters * 2, n_filters * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(n_filters * 4, n_targets, 4, 1, 0, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, input):
        return self.main(input)[:, :, 0, 0]


class GeneratorDCGAN28(nn.Module):
    def __init__(self, in_features, out_channels, n_filters):
        super(GeneratorDCGAN28, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_features, n_filters * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 3, 2, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 7 x 7
            nn.ConvTranspose2d(n_filters * 2, n_filters, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # state size. (ngf*2) x 14 x 14
            nn.ConvTranspose2d(n_filters, out_channels, 3, 2, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, input):
        return self.main(input.view(input.shape[0], input.shape[1], 1, 1))