import torch
import torch.nn as nn
import torch.nn.functional as F

class SNGAN_Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(SNGAN_Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, hidden_size, kernel_size=4, stride=2)),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=4, stride=2)),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(hidden_size * 2, 1, kernel_size=4, stride=2))
        )

    def forward(self, image):
        pred = self.discriminator(image)
        pred = pred.view(pred.size(0), -1)
        return pred
