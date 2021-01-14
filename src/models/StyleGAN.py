import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleGAN_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, z_dim, mapping_hidden_size, w_dim, kernel_size=3, device="cpu"):
        super(StyleGAN_Generator, self).__init__()

        self.device = device

        self.x = nn.Parameter(torch.randn(1, in_channels, 4, 4))
        self.mapping_network = MappingNetwork(z_dim, mapping_hidden_size, w_dim)

        self.block0 = SynthesisNetwork_Block(w_dim, in_channels, hidden_channels, 4, kernel_size, use_upsample=False,
                                             device=self.device)
        self.block1 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 8, kernel_size,
                                             device=self.device)
        self.block2 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 16, kernel_size,
                                             device=self.device)
        self.block3 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 32, kernel_size,
                                             device=self.device)
        self.block4 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 64, kernel_size,
                                             device=self.device)

        self.block1_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block3_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block4_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, z):
        w = self.mapping_network(z)
        x = self.block0(self.x, w)

        x_1 = self.block1(x, w)
        x_2 = self.block2(x_1, w)
        x_3 = self.block3(x_2, w)
        x_4 = self.block4(x_3, w)

        x_4_image = self.block4_to_image(x_4)

        return x_4_image


class StyleGAN_Discriminator(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(StyleGAN_Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            self.get_discriminator_block(in_size, hidden_size),
            self.get_discriminator_block(hidden_size, hidden_size * 2),
            self.get_discriminator_block(hidden_size * 2, hidden_size * 4),
            nn.Conv2d(hidden_size * 4, 1, kernel_size=4, stride=2)
        )

    def forward(self, image):
        preds = self.discriminator(image)
        return preds.view(len(preds), -1)

    def get_discriminator_block(self, in_channels, out_channels, kernel_size=4, stride=2):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        return block


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, hidden_size, w_dim):
        super(MappingNetwork, self).__init__()

        self.mapping_network = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, w_dim)
        )

    def forward(self, z):
        w = self.mapping_network(z)
        return w


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
