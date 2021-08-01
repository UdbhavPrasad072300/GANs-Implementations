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
        return x + self.block


class GlobalGenerator(nn.Module):
    def __init__(self, in_channels, channels, out_channels, num_d_block, num_residual_block, num_u_block):
        super(GlobalGenerator, self).__init__()

        self.G1 = [
            # c7s1-64
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
            nn.Conv2d(in_channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
        ]

        self.G1 = nn.Sequential(*self.G1)

        self.final = nn.Sequential()

    @staticmethod
    def get_D_block(in_channels, out_channels):

        d_block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        ]

        return d_block

    @staticmethod
    def get_U_block(in_channels, out_channels):

        u_block = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        ]

        return u_block

    def forward(self, x):
        return self.final(self.G1(x))


class LocalEnhancer(nn.Module):
    def __init__(self):
        super(LocalEnhancer, self).__init__()

        self.G1 = GlobalGenerator(3, 64, 3, 4, 9, 4)

        self.G2 = nn.Sequential(
            nn.Conv2d(),
        )

    def forward(self, x):
        x = self.G1(x)
        x = self.G2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        return


class MultiscaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiscaleDiscriminator, self).__init__()

    def forward(self):
        return
