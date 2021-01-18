import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleGAN_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, z_dim, mapping_hidden_size, w_dim,
                 synthesis_layers=5, mapping_layers=5, kernel_size=3, device="cpu"):
        super(StyleGAN_Generator, self).__init__()

        self.synthesis_layers = synthesis_layers
        self.mapping_layers = 5
        self.device = device

        self.x = nn.Parameter(torch.randn(1, in_channels, 4, 4))
        self.mapping_network = MappingNetwork(z_dim, mapping_hidden_size, w_dim, self.mapping_layers)

        self.synthesis_network = nn.ModuleList([
            SynthesisNetwork_Block(w_dim, in_channels, hidden_channels, 4, kernel_size,
                                   use_upsample=False, device=self.device)
        ])
        factor = 2
        for layer in range(self.synthesis_layers - 1):
            self.synthesis_network.append(SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 4 * factor,
                                                                 kernel_size, device=self.device))
            factor = factor * 2

        self.block_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, z):
        w = self.mapping_network(z)
        x = self.x

        for layer in self.synthesis_network:
            x = layer(x, w)

        x_image = self.block_to_image(x)

        return x_image


class StyleGAN_Discriminator(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(StyleGAN_Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            self.get_discriminator_block(in_size, hidden_size),
            self.get_discriminator_block(hidden_size, hidden_size * 2),
            self.get_discriminator_block(hidden_size * 2, hidden_size * 4),
            nn.Conv2d(hidden_size * 4, 1, kernel_size=3, stride=2)
        )

    def forward(self, image):
        preds = self.discriminator(image)
        return preds.view(len(preds), -1)

    def get_discriminator_block(self, in_channels, out_channels, kernel_size=3, stride=2):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        return block


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, hidden_size, w_dim, layers=5):
        super(MappingNetwork, self).__init__()

        assert layers > 2, "You need minimum two layers in the mapping network"

        self.layers = layers

        self.mapping_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, hidden_size),
                nn.ReLU(),
            )
        ])
        for layer in range(self.layers):
            self.mapping_network.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                )
            )
        self.mapping_network.append(
            nn.Linear(hidden_size, w_dim)
        )

    def forward(self, z):
        for layer in self.mapping_network:
            z = layer(z)
        return z # or w


class SynthesisNetwork_Block(nn.Module):
    def __init__(self, w_dim, in_channels, out_channels, start_size, kernel_size=3, use_upsample=True, device="cpu"):
        super(SynthesisNetwork_Block, self).__init__()

        self.device = device

        self.use_upsample = use_upsample
        if self.use_upsample:
            self.upsample = nn.Upsample((start_size, start_size), mode="bilinear")

        self.injectnoise = InjectNoise(out_channels, device=self.device)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.adain = AdaIN(w_dim, out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        if self.use_upsample:
            x = self.upsample(x)

        x = self.convolution(x)
        x = self.activation(self.injectnoise(x))

        return self.adain(x, w)


class AdaIN(nn.Module):
    def __init__(self, w_dim, n_channels, device="cpu"):
        super(AdaIN, self).__init__()

        self.device = device

        self.normalize = nn.InstanceNorm2d(n_channels)

        self.style_scale = nn.Linear(w_dim, n_channels)
        self.style_shift = nn.Linear(w_dim, n_channels)

    def forward(self, image, w):
        scale = self.style_scale(w)[:, :, None, None]
        shift = self.style_shift(w)[:, :, None, None]
        return (self.normalize(image) * scale) + shift


class InjectNoise(nn.Module):
    def __init__(self, n_channels, device="cpu"):
        super(InjectNoise, self).__init__()

        self.device = device

        self.weight = nn.Parameter(torch.randn(1, n_channels, 1, 1))

    def forward(self, image):
        return image + (torch.randn(image.size(0), 1, image.size(2), image.size(3)).to(self.device) * self.weight )


class StyleGAN_Discriminator_16x16(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(StyleGAN_Discriminator_16x16, self).__init__()

        self.discriminator = nn.Sequential(
            self.get_discriminator_block(in_size, hidden_size),
            self.get_discriminator_block(hidden_size, hidden_size * 2),
            nn.Conv2d(hidden_size * 2, 1, kernel_size=3, stride=2)
        )

    def forward(self, image):
        preds = self.discriminator(image)
        return preds.view(len(preds), -1)

    def get_discriminator_block(self, in_channels, out_channels, kernel_size=3, stride=2):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        return block