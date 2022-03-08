import sys, math, copy
import numpy as np
import scipy.stats as st

from Code.QualityMeasures import *
from Code.Load import *
from Code.CP import *
from Code.Data import *
from Code.Utils import GaussianLoss, ThresholdAct, apply_dropout

from sklearn.metrics import r2_score
import torch
import torch.nn as nn

def ensemble(model, x, num):
    out = np.zeros((x.shape[0], 2))
    for i in range(num):
        preds = model(x).numpy()
        out[:, 0] += preds[:, 0]
        out[:, 1] += np.exp(preds[:, 1]) + np.square(preds[:, 0])
    out = out / num

    out[:, 1] = np.maximum(0, out[:, 1] - np.square(out[:, 0])) # zero-clipping to avoid NaN results in square root
    return out

def KG(container):

    if container.verbosity() > 0:
        print("Performing KG on", container.dataset(), "for seed", container.seed(), "with dropout optimization")
        print()

    data = init(container.dataset(), container.seed(), cp_mode = container.cp_mode())

    X_train, X_val, y_train, y_val = train_test_split(data["X_train"], data["y_train"], test_size = container.test_frac(), random_state = 2)
    X = torch.from_numpy(X_train).float().requires_grad_(False)
    X_val = torch.from_numpy(X_val).float().requires_grad_(False)
    y = torch.from_numpy(y_train).float().requires_grad_(False)
    y_val = torch.from_numpy(y_val).float().requires_grad_(False)

    best_model = None
    best_loss = 1e8
    train_losses = np.full(container.val_length(), 1e10)
    best_epoch = container.epochs()
    best_cnt = np.Inf
    cnt = 0

    act_threshold = calculateThreshold(data["y_train"]) if container["use_threshold"] else calculateThreshold()

    for drop in container.drop():

        l = (1 - drop) / 2 if container["reg"] else 0

        model = nn.Sequential(
            nn.Linear(X.shape[1], container.dim()),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(container.dim(), container.dim()),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(container.dim(), 2),
            ThresholdAct(act_threshold)
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
        print("Optimized network with", best_epoch + 1, "epochs and dropout", best_model[1])

    cnt = 0

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
            cnt += 1

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
    use_threshold = True, reg = False, **params):

    extra = {"use_threshold": use_threshold, "reg": reg}

    tax = taxonomyFactory(conditional, conditional_params)
    container = NNDataObject(data, "KG", seed = seed, taxonomy_func = tax, loss_func = GaussianLoss, cp_mode = mode, extra = extra, **params)
    container = KG(container)
    container.export(FOLDER)
