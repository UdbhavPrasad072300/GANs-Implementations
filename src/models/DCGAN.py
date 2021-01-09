import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGAN_Generator(nn.Module):
    def __init__(self, noise_size, out_channel, hidden_size):
        super(DCGAN_Generator, self).__init__()

        self.noise_size = noise_size
        self.hidden_size = hidden_size

        self.generator = nn.Sequential(
            self.get_generator_block(noise_size, hidden_size * 4),
            self.get_generator_block(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=1),
            self.get_generator_block(hidden_size * 2, hidden_size),
            nn.ConvTranspose2d(hidden_size, out_channel, kernel_size=4, stride=2),
            nn.Tanh()
        )

    def forward(self, noise):
        noise = noise.view(noise.size(0), self.noise_size, 1, 1)
        return self.generator(noise)

    def get_generator_block(self, in_channels, out_channels, kernel_size=3, stride=2):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        return block

class DCGAN_Discriminator(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(DCGAN_Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            self.get_discriminator_block(in_size, hidden_size),
            self.get_discriminator_block(hidden_size, hidden_size * 2),
            nn.Conv2d(hidden_size * 2, 1, kernel_size=4, stride=2)
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
