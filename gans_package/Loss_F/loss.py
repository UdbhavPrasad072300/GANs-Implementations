import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class W_Gen_Loss(nn.Module):
    def __init__(self):
        super(W_Gen_Loss, self).__init__()

    def forward(self, pred):
        loss = -torch.mean(pred)
        return loss


class W_Crit_Loss(nn.Module):
    def __init__(self, gradient_penalty_weight):
        super(W_Crit_Loss, self).__init__()

        self.c_lambda = gradient_penalty_weight

    def forward(self, fake_pred, real_pred, gradient_penalty):
        loss = torch.mean(fake_pred) - torch.mean(real_pred) + (self.c_lambda * gradient_penalty)
        return loss