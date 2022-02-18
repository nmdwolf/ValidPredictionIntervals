import sys
import numpy as np
import scipy.stats as st

from Code.QualityMeasures import *
from Code.Load import *
from Code.CP import *
from Code.Data import *

import torch
import gpytorch as gp

class MyGP(gp.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean, covar)

def GP(container):

    if container.verbosity() > 0:
        print("Performing GP on", container.dataset(), "for seed", container.seed())
        print()

    data = init(container.dataset(), container.seed(), cp_mode = container.cp_mode())

    X = torch.from_numpy(data["X_train"]).float().requires_grad_(False)
    y = torch.from_numpy(data["y_train"]).float().requires_grad_(False)

    likelihood = gp.likelihoods.FixedNoiseGaussianLikelihood(torch.ones(X.shape[1]) * 0.01, learn_additional_noise = True)
    model = MyGP(X, y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([ \
        {'params': model.parameters()}, \
        ], lr = container.learning_rate())

    # "Loss" for GPs - the marginal log likelihood
    mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(container.epochs()):

        optimizer.zero_grad()

        output = model(X)
        loss = -mll(output, y)
        loss.backward()

        optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad():

        def gp_to_uncertainty(x):
            posterior = likelihood(model(x))
            preds = posterior.mean
            std = torch.sqrt(posterior.variance)
            z = st.norm.ppf(1 - (container.alpha() / 2))
            return torch.stack([preds - z * std, preds + z * std, preds], axis = -1)

        verifier = EnsembleConformalizer(container, gp_to_uncertainty)
        verifier.train(torch.from_numpy(data["X_val"]).float(), data["y_val"])
        verifier.apply(torch.from_numpy(data["X_test"]).float(), data["y_test"])

        if container.taxonomy_func():
            container.taxonomy_func().train(data["X_train"], data["y_train"])
            verifier.train_conditional(torch.from_numpy(data["X_val"]).float(), data["y_val"])
            verifier.apply_conditional(torch.from_numpy(data["X_test"]).float(), data["y_test"])

        return container

def run(data, seed, mode = True, conditional = None, conditional_params = None, **params):

    tax = taxonomyFactory(conditional, conditional_params)
    container = NNDataObject(data, "GP", seed = seed, taxonomy_func = tax, cp_mode = mode, **params)
    container = GP(container)
    container.export(FOLDER)
