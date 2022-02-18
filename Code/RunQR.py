import math, copy, sys
import numpy as np

from Code.QualityMeasures import *
from Code.Load import *
from Code.CP import *
from Code.Data import *

from sklearn.metrics import r2_score
import torch
import torch.nn as nn

def quantile_loss(true, pred, gamma):

    diff = true - pred
    return torch.mean(torch.max((gamma - 1) * diff, gamma * diff))

def pinball_loss(pred, true, quantiles = [0.1, 0.9]):

    # Third index for median !!!
    return torch.mean(torch.stack([quantile_loss(true, pred[:, i], quantiles[i]) for i in range(len(quantiles))]))

# removes crossing incidents
def crossing_act(l):

    a = l[..., 0]
    b = a + nn.ReLU(l[..., 1] - a)

    return torch.stack([a, b], axis = 1)

def QR(container):

    if container.verbosity() > 0:
        print("Performing QR on", container.dataset(), "for seed", container.seed())
        print()

    # TO AVOID OVERLY CONSERVATIVE TRAINING (CP FIXES ANY PROBLEMS)
    if container.cp_mode():
        training_alpha = container["factor"] * container.alpha()
    else:
        training_alpha = container.alpha()
    training_cov = 1 - training_alpha - container["margin"]

    output_dim = 3 if container["median"] else 2
    train_quantiles = [training_alpha / 2, 1 - (training_alpha / 2), 0.5] if container["median"] else [training_alpha / 2, 1 - (training_alpha / 2)]

    data = init(container.dataset(), container.seed(), cp_mode = container.cp_mode())

    X_train, X_val, y_train, y_val = train_test_split(data["X_train"], data["y_train"], test_size = container.test_frac(), random_state = 2)
    X = torch.from_numpy(X_train).float().requires_grad_(False)
    X_val = torch.from_numpy(X_val).float().requires_grad_(False)
    y = torch.from_numpy(y_train).float().requires_grad_(False)
    y_val = torch.from_numpy(y_val).float().requires_grad_(False)

    best_epoch = container.epochs()
    safe_epoch = container.epochs()
    best_cnt = np.Inf
    cnt = 0

    model = nn.Sequential(
        nn.Linear(X.shape[1], container.dim()),
        nn.ReLU(),
        nn.Dropout(container.drop()),
        nn.Linear(container.dim(), container.dim()),
        nn.ReLU(),
        nn.Dropout(container.drop()),
        nn.Linear(container.dim(), output_dim)
    )

    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    m = copy.deepcopy(model)

    optimizer = torch.optim.Adam(model.parameters(), lr = container.learning_rate(), weight_decay = container.l())

    if container.val_length() > 0:

        best_loss = 1e8
        train_covs = np.zeros(container.val_length())
        train_losses = np.full(container.val_length(), 1e10)

        for e in range(container.epochs()):

            model.train()

            shuffle_idx = np.arange(X.shape[0])
            np.random.shuffle(shuffle_idx)
            X = X[shuffle_idx]
            y = y[shuffle_idx]

            for idx in range(0, X.shape[0], container.batch()):

                cnt += 1

                optimizer.zero_grad()

                batch_x = X[idx : min(idx + container.batch(), X.shape[0]),:]
                batch_y = y[idx : min(idx + container.batch(), y.shape[0])]
                preds = model(batch_x)
                loss = container.loss_func(preds, batch_y, train_quantiles)

                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(X_val)
                # loss = container.loss_func(pred, y_val, train_quantiles).numpy()

                train_losses[1:] = train_losses[:-1]
                train_losses[0] = MPIW(pred[:, 0], pred[:, 1], y_val)
                train_covs[1:] = train_covs[:-1]
                train_covs[0] = coverage(pred[:, 0], pred[:, 1], y_val)

                if (np.mean(train_losses) < best_loss) and (np.mean(train_covs) >= training_cov):
                    best_loss = np.mean(train_losses)
                    best_epoch = e
                    best_cnt = cnt

                if np.mean(train_covs) >= container.minimal_cov():
                    safe_epoch = e

        if container.verbosity() > 1:
            print("Optimized network with", best_epoch, "epochs")

    if best_epoch == container.epochs():
        if container.verbosity() > 1:
            print("Defaulting to safe mode with", safe_epoch, "epochs due to low validation coverage")
        best_epoch = safe_epoch

    cnt = 0
    model = m
    optimizer = torch.optim.Adam(model.parameters(), lr = container.learning_rate(), weight_decay = container.l())

    X = torch.from_numpy(data["X_train"]).float().requires_grad_(False)
    y = torch.from_numpy(data["y_train"]).float().requires_grad_(False)

    # for e in range(container.epochs()):
    for e in range(best_epoch + 1):

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
            preds = model(batch_x)
            loss = container.loss_func(preds, batch_y, train_quantiles)

            loss.backward()
            optimizer.step()

            if cnt > best_cnt:
                break

    model.eval()
    with torch.no_grad():
        if not container["median"]:
            m = lambda x: torch.cat([model(x), torch.zeros(x.shape[0]).unsqueeze(-1)], axis = 1)

        verifier = EnsembleConformalizer(container, m)
        verifier.train(torch.from_numpy(data["X_val"]).float(), data["y_val"])
        verifier.apply(torch.from_numpy(data["X_test"]).float(), data["y_test"])

        if container.taxonomy_func():
            container.taxonomy_func().train(data["X_train"], data["y_train"])
            verifier.train_conditional(torch.from_numpy(data["X_val"]).float(), data["y_val"])
            verifier.apply_conditional(torch.from_numpy(data["X_test"]).float(), data["y_test"])

        return container

def run(data, seed, mode = True, conditional = None, conditional_params = None, \
    factor = 2, margin = 0.05, median = True, **params):

    extra = {"factor": factor, "margin": margin, "median": median}

    tax = taxonomyFactory(conditional, conditional_params)
    container = NNDataObject(data, "QR", seed = seed, loss_func = pinball_loss, taxonomy_func = tax, cp_mode = mode, extra = extra, **params)
    container = QR(container)
    container.export(FOLDER, extra = "margin: " + str(margin) + "\nfactor: " + str(factor))
