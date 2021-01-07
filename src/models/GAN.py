import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, noise_size, out_size, hidden_size, num_layers=1):
        super(Generator, self).__init__()

        assert num_layers >= 1

        self.noise_size = noise_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.generator = []

        factor = 1
        for index in range(self.num_layers):
            if index == 0:
                self.generator.append(
                    self.get_generator_block(noise_size, hidden_size)
                )
            else:
                self.generator.append(
                    self.get_generator_block(hidden_size * factor, hidden_size * factor * 2)
                )
                factor = factor * 2

        self.generator = nn.Sequential(
            *self.generator,
            nn.Linear(hidden_size * factor, out_size),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.generator(noise)

    def get_generator_block(self, in_size, out_size):
        block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True)
        )
        return block

class Discriminator(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1):
        super(Discriminator, self).__init__()

        assert num_layers >= 1

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.discriminator = []

        factor = 1
        for index in range(self.num_layers):
            if index == 0:
                self.discriminator.append(
                    self.get_discriminator_block(in_size, hidden_size)
                )
            else:
                self.discriminator.append(
                    self.get_discriminator_block(int(hidden_size / factor), int(hidden_size / (factor * 2)))
                )
                factor = factor * 2

        self.discriminator = nn.Sequential(
            *self.discriminator,
            nn.Linear(int(hidden_size / factor), 1)
        )

    def forward(self, image):
        return self.discriminator(image)

    def get_discriminator_block(self, in_size, out_size, leaky_slope=0.2):
        block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(leaky_slope)
        )
        return block
