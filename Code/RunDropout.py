import sys, math, copy
import numpy as np
import scipy.stats as st

from Code.QualityMeasures import *
from Code.Load import *
from Code.CP import *
from Code.Data import *

import torch
import torch.nn as nn

# Loss function can be derived from MLE with a Guassian distribution N(x^t w, 1).
# L2 penalty can be derived from MAP with Gaussian prior N(0, l^-1) on weights.
def Loss(model, X, true, l = 0):

    preds = model(X)
    loss = torch.mean(torch.square(true - preds)) / 2

    penalty = 0
    for param in model.parameters():
        penalty += torch.norm(param, 2)**2
    penalty *= l / X.shape[0]

    return loss + penalty

def ensemble(model, x, num = 50):
    out = np.zeros((x.shape[0], 2))
    for i in range(num):
        preds = model(x).numpy()
        out[:, 0] = out[:, 0] + preds
        out[:, 1] = out[:, 1] + np.square(preds)
    out = out / num

    out[:, 1] = np.maximum(0, out[:, 1] - np.square(out[:, 0])) # zero-clipping to avoid NaN results in square root
    return out

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def DropoutEnsemble(container):

    if container.verbosity() > 0:
        print("Performing Dropout on", container.dataset(), "for seed", container.seed(), "with dropout optimization")
        print()

    data = init(container.dataset(), container.seed(), cp_mode = container.cp_mode())

    X_train, X_val, y_train, y_val = train_test_split(data["X_train"], data["y_train"], test_size = container.test_frac(), random_state = 2)
    X = torch.from_numpy(X_train).float().requires_grad_(False)
    X_val = torch.from_numpy(X_val).float().requires_grad_(False)
    y = torch.from_numpy(y_train).float().requires_grad_(False)
    y_val = torch.from_numpy(y_val).float().requires_grad_(False)

    best_model = None
    best_epoch = container.epochs()
    best_cnt = np.Inf
    cnt = 0

    if container.val_length() != 0:

        best_loss = 1e8
        train_losses = np.full(container.val_length(), 1e10)

        for drop in container.drop():

            drop = round(drop, 2)
            l = (1 - drop) / 2 if container["reg"] else 0

            model = nn.Sequential(
                nn.Linear(X.shape[1], container.dim()),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(container.dim(), container.dim()),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(container.dim(), 1),
                nn.Flatten(0, 1)
            )

            for m in model:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.constant_(m.bias, 0)

            m = copy.deepcopy(model)

            optimizer = torch.optim.Adam(model.parameters(), lr = container.learning_rate(), weight_decay = 0)
            for e in range(container.epochs()):

                model.train()

                shuffle_idx = np.arange(X.shape[0])
                np.random.shuffle(shuffle_idx)
                X = X[shuffle_idx]
                y = y[shuffle_idx]

                for idx in range(0, X.shape[0], container.batch()):

                    cnt += 1

                    optimizer.zero_grad()

                    batch_x = X[idx : min(idx + container.batch(), X.shape[0]), :]
                    batch_y = y[idx : min(idx + container.batch(), y.shape[0])]
                    loss = container.loss_func(model, batch_x, batch_y, l)

                    loss.backward()
                    optimizer.step()

                model.eval()
                model.apply(apply_dropout)
                with torch.no_grad():
                    loss = container.loss_func(model, X_val, y_val, l).numpy()
                    train_losses[1:] = train_losses[:-1]
                    train_losses[0] = loss

                    if np.mean(train_losses) < best_loss:
                        best_model = (m, drop)
                        best_loss = np.mean(train_losses)
                        best_epoch = e
                        best_cnt = cnt

        if container.verbosity() > 1:
            print("Optimized network with", best_epoch, "epochs and dropout", best_model[1])

    else:
        print("No early stopping performed. Default dropout 0.1 used.")
        best_model = (nn.Sequential(
            nn.Linear(X.shape[1], container.dim()),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(container.dim(), container.dim()),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(container.dim(), 1),
            nn.Flatten(0, 1)
        ), 0.1)

    cnt = 0
    best_cnt = best_cnt if container.val_length() > 0 else np.Inf

    model = best_model[0]
    optimizer = torch.optim.Adam(model.parameters(), lr = container.learning_rate(), weight_decay = 0)
    model.train()

    X = torch.from_numpy(data["X_train"]).float().requires_grad_(False)
    y = torch.from_numpy(data["y_train"]).float().requires_grad_(False)

    for e in range(best_epoch + 1):

        if cnt > best_cnt:
            break

        shuffle_idx = np.arange(X.shape[0])
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        for idx in range(0, X.shape[0], container.batch()):

            if cnt > best_cnt:
                break
            cnt = cnt + 1

            optimizer.zero_grad()

            batch_x = X[idx : min(idx + container.batch(), X.shape[0]),:]
            batch_y = y[idx : min(idx + container.batch(), y.shape[0])]
            loss = container.loss_func(model, batch_x, batch_y, l)

            loss.backward()
            optimizer.step()

    model.eval()
    model.apply(apply_dropout)
    with torch.no_grad():

        def ensemble_to_quantile(x):
            preds = ensemble(model, x, container.num())
            sig = np.sqrt(preds[:, 1])
            z = st.norm.ppf((1 - container.alpha()) + container.alpha() / 2)
            lower = preds[:, 0] - z * sig
            upper = preds[:, 0] + z * sig
            return np.stack([lower, upper, preds[:, 0]], axis = -1)

        verifier = EnsembleConformalizer(container, ensemble_to_quantile)
        verifier.train(torch.from_numpy(data["X_val"]).float(), data["y_val"])
        verifier.apply(torch.from_numpy(data["X_test"]).float(), data["y_test"])

        if container.taxonomy_func():
            container.taxonomy_func().train(data["X_train"], data["y_train"])
            verifier.train_conditional(torch.from_numpy(data["X_val"]).float(), data["y_val"])
            verifier.apply_conditional(torch.from_numpy(data["X_test"]).float(), data["y_test"])

        return container

def run(data, seed, mode = True, conditional = None, conditional_params = None, \
    reg = False, **params):

    extra = {"reg": reg}

    tax = taxonomyFactory(conditional, conditional_params)
    container = NNDataObject(data, "Dropout", seed = seed, taxonomy_func = tax, loss_func = Loss, cp_mode = mode, extra = extra, **params)
    container = DropoutEnsemble(container)
    container.export(FOLDER)
