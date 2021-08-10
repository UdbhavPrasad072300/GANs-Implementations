import torch
import torch.nn as nn
import torch.nn.functional as F


class TransGAN_Generator(nn.Module):
    def __init__(self):
        super(TransGAN_Generator, self).__init__()

    def forward(self, x):
        return x


class ViT_Discriminator(nn.Module):
    def __init__(self):
        super(ViT_Discriminator, self).__init__()

    def forward(self, x):
        return x
