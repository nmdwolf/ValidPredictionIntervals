import copy, sys
import numpy as np

from Code.CP import *
from Code.Data import *
from Code.Load import *
from Code.QualityMeasures import *

import torch
import torch.nn as nn

def NNCP(container):

    if container.verbosity() > 0:
        print("Performing Conformalized NN on", container.dataset(), "for seed", container.seed())
        print()

    data = init(container.dataset(), container.seed())

    X_train, X_val, y_train, y_val = train_test_split(data["X_train"], data["y_train"], test_size = container.test_frac(), random_state = 2)
    X = torch.from_numpy(X_train).float().requires_grad_(False)
    X_val = torch.from_numpy(X_val).float().requires_grad_(False)
    y = torch.from_numpy(y_train).float().requires_grad_(False)
    y_val = torch.from_numpy(y_val).float().requires_grad_(False)

    best_epoch = container.epochs()
    best_cnt = np.Inf
    cnt = 0

    model = nn.Sequential(
        nn.Linear(X.shape[1], container.dim()),
        nn.ReLU(),
        nn.Dropout(container.drop()),
        nn.Linear(container.dim(), container.dim()),
        nn.ReLU(),
        nn.Dropout(container.drop()),
        nn.Linear(container.dim(), 1),
        nn.Flatten(0, 1)
    )

    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    m = copy.deepcopy(model)

    best_loss = 1e8
    train_losses = np.full(container.val_length(), 1e10)

    optimizer = torch.optim.Adam(model.parameters(), lr = container.learning_rate(), weight_decay = container.l())

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
            loss = container.loss_func(model(batch_x), batch_y)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss = container.loss_func(model(X_val), y_val).numpy()
            train_losses[1:] = train_losses[:-1]
            train_losses[0] = loss

        if np.mean(train_losses) < best_loss:
            best_loss = np.mean(train_losses)
            best_epoch = e
            best_cnt = cnt

    if container.verbosity() > 1:
        print("Optimized network with", best_epoch + 1, "epochs")

    cnt = 0
    model = m
    optimizer = torch.optim.Adam(model.parameters(), lr = container.learning_rate(), weight_decay = container.l())
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
            cnt += 1

            optimizer.zero_grad()

            batch_x = X[idx : min(idx + container.batch(), X.shape[0]),:]
            batch_y = y[idx : min(idx + container.batch(), y.shape[0])]
            loss = container.loss_func(model(batch_x), batch_y)

            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():

        verifier = PointConformalizer(container, model)
        verifier.train(torch.from_numpy(data["X_val"]).float(), data["y_val"])
        verifier.apply(torch.from_numpy(data["X_test"]).float(), data["y_test"])

        if container.taxonomy_func():
            container.taxonomy_func().train(data["X_train"], data["y_train"])
            verifier.train_conditional(torch.from_numpy(data["X_val"]).float(), data["y_val"])
            verifier.apply_conditional(torch.from_numpy(data["X_test"]).float(), data["y_test"])

        return container

def run(data, seed, mode = True, conditional = None, conditional_params = None, **params):

    tax = taxonomyFactory(conditional, conditional_params)
    container = NNDataObject(data, "NN", seed = seed, taxonomy_func = tax, **params)
    container = NNCP(container)
    container.export(FOLDER)
