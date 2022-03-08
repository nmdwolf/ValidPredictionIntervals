import sys, math, copy
import numpy as np
import scipy.stats as st

from Code.QualityMeasures import *
from Code.Load import *
from Code.CP import *
from Code.Data import *
from Code.Utils import GaussianLoss, ThresholdAct

import torch
import torch.nn as nn

# Fast Gradient Sign method for adversarial pertubations
def perturb(model, x, y, epsilon = 0.1):
    delta = torch.zeros_like(x, requires_grad = True)
    loss = GaussianLoss(model, x + delta, y)
    loss.backward()
    return x + epsilon * delta.grad.detach().sign()

def ensemble(models, x):
    out = np.zeros((x.shape[0], 2))
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = model(x).numpy()
            out[:, 0] += preds[:, 0]
            out[:, 1] += np.exp(preds[:, 1]) + np.square(preds[:, 0])
    out = out / len(models)
    out[:, 1] = np.maximum(0, out[:, 1] - np.square(out[:, 0])) # zero-clipping to avoid NaN results in square root
    return out

def DE(container):

    if container.verbosity() > 0:
        print("Performing DE on", container.dataset(), "for seed", container.seed())
        print()

    data = init(container.dataset(), container.seed(), cp_mode = container.cp_mode())

    # Variable epsilon based on range of each feature
    epsilon = container["step"] * (np.amax(data["X_train"], axis = 0) - np.amin(data["X_train"], axis = 0))
    epsilon = torch.from_numpy(epsilon).float().requires_grad_(False)

    act_threshold = calculateThreshold(data["y_train"]) if container["use_threshold"] else calculateThreshold()

    model_list = []
    for i in range(container.num()):

        X_train, X_val, y_train, y_val = train_test_split(data["X_train"], data["y_train"], test_size = container.test_frac(), random_state = 2)
        X = torch.from_numpy(X_train).float().requires_grad_(False)
        X_val = torch.from_numpy(X_val).float().requires_grad_(False)
        y = torch.from_numpy(y_train).float().requires_grad_(False)
        y_val = torch.from_numpy(y_val).float().requires_grad_(False)

        best_loss = 1e8
        train_losses = np.full(container.val_length(), 1e10)
        best_epoch = container.epochs()
        best_cnt = np.Inf
        cnt = 0

        model = nn.Sequential(
            nn.Linear(X.shape[1], container.dim()),
            nn.ReLU(),
            nn.Linear(container.dim(), container.dim()),
            nn.ReLU(),
            nn.Linear(container.dim(), 2),
            ThresholdAct(act_threshold)
        )

        for m in model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

        m = copy.deepcopy(model)

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

                perturbed_x = perturb(model, batch_x, batch_y, epsilon = epsilon)
                optimizer.zero_grad()

                loss = container.loss_func(model, batch_x, batch_y) + container.loss_func(model, perturbed_x, batch_y)
                loss.backward()

                optimizer.step()

            model.eval()
            perturbed_x = perturb(model, X_val, y_val, epsilon = epsilon)
            with torch.no_grad():
                loss = container.loss_func(model, X_val, y_val) + container.loss_func(model, perturbed_x, y_val)
                loss = loss.numpy()

                train_losses[1:] = train_losses[:-1]
                train_losses[0] = loss

                if np.mean(train_losses) < best_loss:
                    best_loss = np.mean(train_losses)
                    best_epoch = e
                    best_cnt = cnt

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

                perturbed_x = perturb(model, batch_x, batch_y, epsilon = epsilon)
                optimizer.zero_grad()

                loss = container.loss_func(model, batch_x, batch_y) + container.loss_func(model, perturbed_x, batch_y)
                loss.backward()

                optimizer.step()

        model_list.append(model)

        if container.verbosity() > 1:
            print("Optimized network", i + 1, " with", best_epoch + 1, "epochs")

    def ensemble_to_quantile(x):
        preds = ensemble(model_list, x)
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
    adversarial_step = 0.01, threshold = False, **params):

    extra = {"use_threshold": threshold, "step": adversarial_step}

    tax = taxonomyFactory(conditional, conditional_params)
    container = NNDataObject(data, "DE", seed = seed, taxonomy_func = tax, loss_func = GaussianLoss, cp_mode = mode, extra = extra, **params)
    container = DE(container)
    container.export(FOLDER)
