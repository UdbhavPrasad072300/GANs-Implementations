import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGAN_Generator(nn.Module):
    def __init__(self, noise_size, out_channel, hidden_size, num_layers=1, kernel_size=3, stride=2, factor=1):
        super(DCGAN_Generator, self).__init__()

        assert num_layers >= 1

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.generator = []

        for index in range(self.num_layers):
            if index==0:
                self.generator.append(
                    self.get_generator_block(noise_size, hidden_size, kernel_size=kernel_size, stride=stride)
                )
            else:
                self.generator.append(
                    self.get_generator_block(int(hidden_size / factor), int(hidden_size / (factor * 2)), kernel_size=kernel_size,
                                             stride=stride)
                )
                factor = factor * 2

        self.generator = nn.Sequential(
            *self.generator,
            nn.ConvTranspose2d(int(hidden_size / factor), out_channel, kernel_size=kernel_size, stride=stride),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.generator(noise)

    def get_generator_block(self, in_channels, out_channels, kernel_size=3, stride=2):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

class DCGAN_Discriminator(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, kernel_size=4, stride=2, factor=1):
        super(DCGAN_Discriminator, self).__init__()

        assert num_layers >= 1

        self.num_layers = num_layers

        self.discriminator = []

        for index in range(self.num_layers):
            if index==0:
                self.discriminator.append(
                    self.get_discriminator_block(in_size, hidden_size, kernel_size, stride)
                )
            else:
                self.discriminator.append(
                    self.get_discriminator_block(hidden_size * factor, hidden_size * factor * 2, kernel_size, stride)
                )
                factor = factor * 2

        self.discriminator = nn.Sequential(
            *self.discriminator,
            nn.Conv2d(hidden_size * factor, 1, kernel_size=kernel_size, stride=stride)
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
