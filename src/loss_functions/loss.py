import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Loss():
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self):
        return