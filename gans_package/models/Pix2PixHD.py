import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3),
            nn.InstanceNorm2d(n_channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3),
            nn.InstanceNorm2d(n_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class GlobalGenerator(nn.Module):
    def __init__(self, in_channels, channels, out_channels, num_d_block, num_residual_block, num_u_block):
        super(GlobalGenerator, self).__init__()

        # c7s1-64
        self.G1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, channels, kernel_size=7),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
        ]

        # d128, d256, d512, d1024
        for _ in range(num_d_block):
            self.G1 += self.get_D_block(channels, channels*2)
            channels *= 2

        # R1024, R1024, R1024, R1024, R1024, R1024, R1024, R1024, R1024
        self.G1 += [ResidualBlock(channels) for _ in range(num_residual_block)]

        # u512,u256,u128,u64
        for _ in range(num_u_block):
            self.G1 += self.get_U_block(channels, channels // 2)
            channels //= 2

        # c7s1-3
        self.G1 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
        ]

        self.G1 = nn.Sequential(*self.G1)

        self.final = nn.Sequential()

    @staticmethod
    def get_D_block(in_channels, out_channels):

        d_block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        ]

        return d_block

    @staticmethod
    def get_U_block(in_channels, out_channels):

        u_block = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        ]

        return u_block

    def forward(self, x):
        return self.final(self.G1(x))


class LocalEnhancer(nn.Module):
    def __init__(self):
        super(LocalEnhancer, self).__init__()

        self.down_sample_layer = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

        self.G1 = GlobalGenerator(3, 64, 3, 4, 9, 4)

        self.G2_1 = nn.Sequential(
            nn.Conv2d(),
        )

        self.G2_2 = nn.Sequential(
            nn.Conv2d(),
        )

    def forward(self, x):
        x_down_sample = self.down_sample_layer(x)
        x_G1 = self.G1(x_down_sample)
        x_G2_1 = self.G2_1(x)
        return self.G2_2(x_G2_1 + x_G1)


class Discriminator(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Discriminator, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                # C64
                nn.Conv2d(in_channels, channels, kernel_size=4, stride=2),
                nn.InstanceNorm2d(channels),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                # C128
                nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2),
                nn.InstanceNorm2d(channels * 2),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                # C256
                nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2),
                nn.InstanceNorm2d(channels * 4),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                # C512
                nn.Conv2d(channels * 4, channels * 8, kernel_size=4, stride=2),
                nn.InstanceNorm2d(channels * 8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(channels * 8, out_channels, kernel_size=4)
            ),
        ])

    def forward(self, x):
        features = []
        for layer in self.layers():
            x = layer(x)
            features.append(x)
        return features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_discriminators):
        super(MultiScaleDiscriminator, self).__init__()

        self.down_sample_layer = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

        self.discriminators = nn.ModuleList([Discriminator(3, 64, 3) for _ in n_discriminators])

    def forward(self, x):
        for discriminator in self.discriminators:
            x = discriminator(x)
        return
