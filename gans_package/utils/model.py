import torch
import torch.nn as nn


def weights_init(m, mean=0.0, standard_deviation=0.02):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean, standard_deviation)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean, standard_deviation)
        torch.nn.init.constant_(m.bias, 0)
