import sys

from Code.QualityMeasures import *
from Code.Load import *
from Code.CP import *
from Code.Data import *

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def RFCP(container):

    if container.verbosity() > 0:
        print("Performing Conformalized RF on", container.dataset(), "for seed", container.seed())
        print()

    data = init(container.dataset(), container.seed())

    model = RandomForestRegressor(verbose = 0, n_estimators = container.trees())
    model.fit(data["X_train"], data["y_train"])

    verifier = PointConformalizer(container, model.predict)
    verifier.train(torch.from_numpy(data["X_val"]).float(), data["y_val"])
    verifier.apply(torch.from_numpy(data["X_test"]).float(), data["y_test"])

    if container.taxonomy_func():
        container.taxonomy_func().train(data["X_train"], data["y_train"])
        verifier.train_conditional(torch.from_numpy(data["X_val"]).float(), data["y_val"])
        verifier.apply_conditional(torch.from_numpy(data["X_test"]).float(), data["y_test"])

    return container

def run(data, seed, mode = True, conditional = None, conditional_params = None, **params):

    tax = taxonomyFactory(conditional, conditional_params)
    container = ForestDataObject(data, "RF", seed = seed, taxonomy_func = tax, **params)
    container = RFCP(container)
    container.export(FOLDER)
