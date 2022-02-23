import numpy as np

import torch
import torch.nn as nn

def quantile_loss(true, pred, gamma):

    diff = true - pred
    return torch.mean(torch.max((gamma - 1) * diff, gamma * diff))

def pinball_loss(pred, true, quantiles = [0.1, 0.9]):

    # Third index for median !!!
    return torch.mean(torch.stack([quantile_loss(true, pred[:, i], quantiles[i]) for i in range(len(quantiles))]))

# Loss function can be derived from MLE with a Guassian distribution N(x^t w, 1).
# L2 penalty can be derived from MAP with Gaussian prior N(0, l^-1) on weights.
def L2Loss(model, X, true, l = 0):

    preds = model(X)
    loss = torch.mean(torch.square(true - preds)) / 2

    penalty = 0
    for param in model.parameters():
        penalty += torch.norm(param, 2)**2
    penalty *= l / X.shape[0]

    return loss + penalty

def GaussianLoss(model, X, true, l = 0, regularization = 1e-8):

    preds = model(X)
    mse = torch.square(true - preds[:, 0])
    var = torch.exp(preds[:, 1])
    loss = torch.mean(mse / (var + regularization) + preds[:, 1])

    penalty = 0
    for param in model.parameters():
        penalty += torch.norm(param, 2) ** 2
    penalty *= l / X.shape[0]

    return loss + penalty

# removes crossing incidents
def crossing_act(l):

    a = l[..., 0]
    b = a + nn.ReLU(l[..., 1] - a)

    return torch.stack([a, b], axis = 1)

class ThresholdAct(torch.nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, l):
        return torch.stack([l[:, 0], -torch.nn.Threshold(self.threshold, self.threshold)(-l[:, 1])], dim = 1)

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
