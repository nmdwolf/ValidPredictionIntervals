import sys
import numpy as np
import scipy.stats as st

from Code.QualityMeasures import *
from Code.Load import *
from Code.CP import *
from Code.Data import *

import torch
import gpytorch as gp
from torch.utils.data import TensorDataset, DataLoader

class MyGP(gp.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gp.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gp.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean, covar)

def AGP(container):

    if container.verbosity() > 0:
        print("Performing GP on", container.dataset(), "for seed", container.seed())
        print()

    data = init(container.dataset(), container.seed(), cp_mode = container.cp_mode())

    X = torch.from_numpy(data["X_train"]).float().requires_grad_(False)
    y = torch.from_numpy(data["y_train"]).float().requires_grad_(False)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size = container.batch())

    inducing_points = X[:500, :]
    likelihood = gp.likelihoods.FixedNoiseGaussianLikelihood(torch.ones(X.shape[1]) * 0.01, learn_additional_noise = True)
    model = MyGP(inducing_points)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([ \
        {'params': model.parameters()}, \
        {'params': likelihood.parameters()}, \
        ], lr = container.learning_rate())

    # "Loss" for GPs - the marginal log likelihood
    mll = gp.mlls.VariationalELBO(likelihood, model, num_data = y.size(0))

    for i in range(container.epochs()):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            output = model(batch_x)
            loss = -mll(output, batch_y)
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
    container = NNDataObject(data, "AGP", seed = seed, taxonomy_func = tax, cp_mode = mode, **params)
    container = AGP(container)
    container.export(FOLDER)
