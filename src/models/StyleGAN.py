import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleGAN_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, z_dim, mapping_hidden_size, w_dim, kernel_size=3):
        super(StyleGAN_Generator, self).__init__()

        self.x = nn.Parameter(torch.randn(1, in_channels, 4, 4))
        self.mapping_network = MappingNetwork(z_dim, mapping_hidden_size, w_dim)

        self.block0 = SynthesisNetwork_Block(w_dim, in_channels, hidden_channels, 4, kernel_size, use_upsample=False)
        self.block1 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 8, kernel_size)
        self.block2 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 16, kernel_size)
        self.block3 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 32, kernel_size)
        self.block4 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 64, kernel_size)
        self.block6 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 128, kernel_size)
        self.block7 = SynthesisNetwork_Block(w_dim, hidden_channels, hidden_channels, 256, kernel_size)

        self.block1_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block3_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block4_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block5_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block6_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.block7_to_image = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, z):
        w = self.mapping_network(z)
        x = self.block0(self.x, w)

        x_1 = self.block1(x, w)
        x_2 = self.block2(x, w)
        x_3 = self.block3(x, w)
        x_4 = self.block4(x, w)
        x_5 = self.block5(x, w)
        x_6 = self.block6(x, w)
        x_7 = self.block7(x, w)

        x_7_image = self.block7_to_image(x_7)

        return x_7_image


class StyleGAN_Discriminator(nn.Module):
    def __init__(self):
        super(StyleGAN_Discriminator, self).__init__()

    def forward(self, image):
        return


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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, w_dim)
        )

    def forward(self, z):
        w = self.mapping_network(z)
        return


class SynthesisNetwork_Block(nn.Module):
    def __init__(self, w_dim, in_channels, out_channels, start_size, kernel_size=3, use_upsample=True):
        super(SynthesisNetwork_Block, self).__init__()

        self.use_upsample = use_upsample
        if self.use_upsample:
            self.upsample = nn.Upsample((start_size, start_size), mode="bilinear")

        self.injectnoise = InjectNoise(out_channels)
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
    def __init__(self, w_dim, n_channels):
        super(AdaIN, self).__init__()

        self.normalize = nn.InstanceNorm2d(n_channels)

        self.style_scale = nn.Linear(w_dim, n_channels)
        self.style_shift = nn.Linear(w_dim, n_channels)

    def forward(self, image, w):
        scale = self.style_scale(w)[:, :, None, None]
        shift = self.style_shift(w)[:, :, None, None]
        return (self.normalize(image) * scale) + shift


class InjectNoise(nn.Module):
    def __init__(self, n_channels):
        super(InjectNoise, self).__init__()

        self.weight = nn.Parameter(torch.randn(1, n_channels, 1, 1))

    def forward(self, image):
        return image + (torch.randn(image.size(0), 1, image.size(2), image.size(3)) * self.weight )
