import math
import numpy as np
import torch

# helper function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# helper function to convert Torch to NumPy
def to_numpy(lowers, uppers, true):
    if torch.is_tensor(lowers):
        lowers = lowers.numpy().squeeze()
    if torch.is_tensor(uppers):
        uppers = uppers.numpy().squeeze()
    if torch.is_tensor(true):
        true = true.numpy().squeeze()

    return lowers, uppers, true

def coverage(lowers, uppers, true):

    # coverage = 0
    # for i in range(len(true)):
    #     if true[i] >= lowers[i] and true[i] <= uppers[i]:
    #         coverage += 1
    # PICP = coverage / len(true)
    #
    # return PICP

    lowers, uppers, true = to_numpy(lowers, uppers, true)
    return np.sum((true >= lowers) & (true <= uppers)) / len(true)

def MPIW(lowers, uppers, true):
    lowers, uppers, true = to_numpy(lowers, uppers, true)
    return np.mean(uppers - lowers)

def MPIWabs(lowers, uppers, true):
    lowers, uppers, true = to_numpy(lowers, uppers, true)
    return np.mean(np.abs(uppers - lowers))

def pinball(pred, true, quantiles = [0.05, 0.95]):

    def intern(true, pred, gamma):
        diff = true - pred
        return np.mean(np.maximum((gamma - 1) * diff, gamma * diff))

    return np.mean([intern(true, pred[:, i], quantiles[i]) for i in range(len(quantiles))])

def HSMAPE(true, pred):
    mu = np.mean(true)
    return np.mean(np.abs(true - pred)) / mu

# Based on paper by Khosravi and Creighton
def CWC(lowers, uppers, true, alpha = 0.05, eta = 50):
    mu = 1 - alpha

    coverage = 0
    MPIW = 0
    for i in range(len(true)):
        MPIW = MPIW + (uppers[i] - lowers[i])
        if true[i] >= lowers[i] and true[i] <= uppers[i]:
            coverage = coverage + 1
    PICP = coverage / len(true)
    MPIW = MPIW / len(true)
    NMPIW = MPIW / (np.max(true) - np.min(true))

    print("Coverage probability ", PICP)
    print("MPIW (/N): ", MPIW, NMPIW)

    gamma = 1 if PICP < mu else 0

    return NMPIW * (1 + gamma * math.exp(-eta * (PICP - mu)))

def QD(lowers, uppers, true, alpha = 0.05, rho = 10):
    dim = len(true)
    empty = np.zeros(dim)
    ups = np.maximum(empty, np.sign(uppers - true))
    lows = np.maximum(empty, np.sign(true - lowers))
    res = lows * ups

    cMPIW = np.sum((uppers - lowers) * res) / (np.sum(res) + 1e-8)
    PICP = np.mean(res)

    qd = cMPIW + rho * (dim / (alpha * (1 - alpha))) * np.square(np.maximum(0., (1. - alpha) - PICP))
    return qd

def QDs(lowers, uppers, true, alpha = 0.05, rho = 10, soft = 50):

    dim = len(true)
    empty = np.zeros(dim)
    ups = np.maximum(empty, np.sign(uppers - true))
    lows = np.maximum(empty, np.sign(true - lowers))
    res = lows * ups

    cMPIW = np.sum((uppers - lowers) * res) / (np.sum(res) + 1e-8)

    u = sigmoid(soft * (uppers - true))
    l = sigmoid(soft * (true - lowers))
    r = l * u

    PICP = np.mean(r)

    loss = cMPIW + rho * (dim / (alpha * (1 - alpha))) * np.square(np.maximum(0., (1. - alpha) - PICP))
    return loss

def QDs_measure(lowers, uppers, true, alpha = 0.05, rho = 10, soft = 50):

    cMPIW = np.sum(uppers - lowers) / len(true)

    u = sigmoid(soft * (uppers - true))
    l = sigmoid(soft * (true - lowers))
    r = l * u

    PICP = np.mean(r)

    loss = cMPIW + rho * (len(true) / (alpha * (1 - alpha))) * np.square(np.maximum(0., (1. - alpha) - PICP))
    return loss
