from Code.QualityMeasures import *

from scipy.optimize import root_scalar

import math
import numpy as np
import warnings

__CONDITIONAL_MIN__ = 2


# helper function for the calculation of percentiles:
def empirical_percentile(data, quantile):
    sorted = np.sort(data)
    index = math.ceil(quantile * (1 + len(sorted))) - 1 # index corresponding to inflated quantile
    return sorted[min(max(0, index), len(sorted) - 1)]

############################
############################



def DISTscore(model, val_X, val_y):
    preds = model(val_X).squeeze()
    return np.abs(preds - val_y)

def QRscore(model, val_X, val_y):
    preds = model(val_X)
    if torch.is_tensor(preds):
        preds = preds.numpy()
    preds = preds.squeeze()
    return np.maximum(preds[:, 0] - val_y, val_y - preds[:, 1])

# General (inductive) conformalization method
#
# Arguments:
#   f: method that computes conformity scores
#   val_X: feature matrix (validation set)
#   val_y: response variable (validation set)
#   alpha: significance
def CP(f, val_X, val_y, alpha = 0.1):
    score = f(val_X, val_y)
    return empirical_percentile(score, 1 - alpha)

################################################################################
################################################################################
#
#   Conditional CP (not actively used yet !!!!)
#
################################################################################
################################################################################

# Abstract class representing a taxonomy function
class Taxonomy:

    def __init__(self, num):
        self.num = num

    def __str__(self):
        return "AbstractTaxonomy-"+str(self.num)

    def train(self, X, y):
        pass

    def classify(self, X, y):
        pass

    def split_count(self):
        return self.num

class MultisplitTaxonomy(Taxonomy):

    def __init__(self, num):
        if num <= 1:
            raise Exception("Number of classes should be > 1")
        super().__init__(num)
        self.splits = np.zeros(self.num - 1)

    def __str__(self):
        return "MultisplitTaxonomy-"+str(self.num)

    def train(self, X, y):
        sorted = np.sort(y)
        self.splits = [sorted[math.floor(len(sorted) / self.num) * i] for i in range(1, self.num)]

    def classify(self, X, y):
        return np.array([self.num - np.sum(self.splits > val) for val in y])

class MultiquantileTaxonomy(Taxonomy):

    def __init__(self, quantiles):
        if len(quantiles) < 1:
            raise Exception("Number of classes should be > 1")
        super().__init__(len(quantiles) + 1)
        self.splits = np.zeros(self.num - 1)
        self.__quantiles = quantiles

    def __str__(self):
        return "MultiquantileTaxonomy-"+str(self.__quantiles)

    def train(self, X, y):
        self.splits = np.quantile(y, self.__quantiles)

    def classify(self, X, y):
        return np.array([self.num - np.sum(self.splits > val) for val in y])

def taxonomyFactory(type, params):
    if type == "multisplit" or type == "split":
        return MultisplitTaxonomy(int(params))
    elif type == "multiquantile" or type == "quantile" or type == "quant":
        return MultiquantileTaxonomy([float(p) for p in params])

    return None

# Conditional (inductive) conformalization method
#
# Arguments:
#   f: method that computes conformity scores
#   tax: taxonomy method that classifies instances based upon a given rule
#   val_X: feature matrix (validation set)
#   val_y: response variable (validation set)
#   alpha: significance
def conditionalCP(f, tax, val_X, val_y, alpha = 0.1, suppress_verbosity = False):
    score = f(val_X, val_y)
    taxonomy = tax.classify(val_X, val_y)

    thresholds = dict()
    for t in range(tax.split_count() + 1):
        thresholds[t] = 0
        if t in np.unique(taxonomy):
            score_t = score[taxonomy == t]

            #########
            if not suppress_verbosity:
                print("class", t, "size", len(score_t))
            #########

            if len(score_t) == 0:
                warnings.warn("Class "+str(t)+" contains 0 instances. It might be better to use different splits. Defaulting to threshold 0.")
            else:
                thresholds[t] = empirical_percentile(score_t, 1 - alpha)

    return thresholds

################################################################################
################################################################################
#
#   Application
#
################################################################################
################################################################################

class Conformalizer():

    def __init__(self, container, model, train_model, disconformity_score):
        self.container = container
        self.model = model
        self.train_model = model if not train_model else train_model
        self.__score = disconformity_score

        self.threshold = np.Inf
        if self.container.taxonomy_func():
            self.conditional_thresholds = dict()
            for i in range(1, self.container.taxonomy_func().split_count() + 1):
                self.conditional_thresholds[i] = np.Inf

    def train(self, x, y, suppress_verbosity = False):
        if self.container.cp_mode():
            self.threshold = empirical_percentile(self.__score(self.train_model, x, y), 1 - self.container.alpha())
            if self.container.verbosity() > 1 and not suppress_verbosity:
                print("Conformalized with threshold:", self.threshold)
                print()

    # NOT USED YET
    def train_conditional(self, x, y, suppress_verbosity = False):
        if self.container.cp_mode():
            self.conditional_thresholds.update(conditionalCP(lambda x, y: self.__score(self.train_model, x, y), self.container.taxonomy_func(), x, y, self.container.alpha(), suppress_verbosity))

    def apply(self, x, y):
        pass

    def apply_conditional(self, x, y):
        pass

class PointConformalizer(Conformalizer):

    def __init__(self, container, model, train_model = None):
        super().__init__(container, model, train_model, DISTscore)

    def apply(self, x, y):

        preds = self.model(x)
        if torch.is_tensor(preds):
            preds = preds.numpy()
        preds = preds.squeeze()

        lower = preds - self.threshold
        upper = preds + self.threshold

        self.container["cp_cov"] = coverage(lower, upper, y)
        self.container["cp_width"] = MPIW(lower, upper, y)
        self.container["cp_perf"] = self.container.perf_func(y, preds)

        if self.container.verbosity() > 0:
            print()
            print("Coverage:", self.container["cp_cov"])
            print("Average width:", self.container["cp_width"])
            print("Performance:", self.container["cp_perf"])
            print()

    # NOT USED YET
    def apply_conditional(self, x, y):

        preds = self.model(x)
        if torch.is_tensor(preds):
            preds = preds.numpy()
        preds = preds.squeeze()

        taxonomy = self.container.taxonomy_func().classify(x, y)
        threshold = np.array([self.conditional_thresholds[taxonomy[i]] for i in range(len(taxonomy))])

        lower = preds - threshold
        upper = preds + threshold

        for t in sorted(self.conditional_thresholds.keys()):
            index = (taxonomy == t)
            if np.sum(index) > __CONDITIONAL_MIN__:
                self.container["conditional_cp_cov", t] = coverage(lower[index], upper[index], y[index])
                self.container["conditional_cp_width", t] = MPIW(lower[index], upper[index], y[index])
                self.container["conditional_cp_perf", t] = self.container.perf_func(y[index], preds[index])

class EnsembleConformalizer(Conformalizer):

    # Arguments:
    #   ensemble_function: Function that takes a list of models and creates an ensemble prediction.
    #                      Should output a NumPy array of the form [lower_quantiles, upper_quantiles, predictions]
    def __init__(self, container, ensemble_func, train_ensemble_func = None):
        super().__init__(container, ensemble_func, train_ensemble_func, QRscore)

    def apply(self, x, y):

        preds = self.model(x)
        if torch.is_tensor(preds):
            preds = preds.numpy()
        preds = preds.squeeze()

        lower = preds[:, 0]
        upper = preds[:, 1]
        preds = preds[:, 2]

        # Check performance WITHOUT conformalization
        self.container["cov"] = coverage(lower, upper, y)
        self.container["width"] = MPIW(lower, upper, y)
        self.container["perf"] = self.container.perf_func(y, preds)

        if self.container.verbosity() > 0:
            print()
            print("Coverage:", self.container["cov"])
            print("Average width:", self.container["width"])
            print("Performance:", self.container["perf"])
            print()

        # Check performance WITH conformalization
        if self.container.cp_mode():
            lower -= self.threshold
            upper += self.threshold

            self.container["cp_cov"] = coverage(lower, upper, y)
            self.container["cp_width"] = MPIW(lower, upper, y)
            self.container["cp_perf"] = self.container.perf_func(y, preds)

            if self.container.verbosity() > 0:
                print()
                print("Conformalized Coverage:", self.container["cp_cov"])
                print("Conformalized Average width:", self.container["cp_width"])
                print("Conformalized Performance:", self.container["cp_perf"])
                print()

    # NOT USED YET
    def apply_conditional(self, x, y):

        preds = self.model(x)
        if torch.is_tensor(preds):
            preds = preds.numpy()
        preds = preds.squeeze()

        lower = preds[:, 0]
        upper = preds[:, 1]
        preds = preds[:, 2]

        taxonomy = self.container.taxonomy_func().classify(x, y)

        # Check performance WITHOUT conformalization
        for t in sorted(self.conditional_thresholds.keys()):
            index = (taxonomy == t)
            if np.sum(index) > __CONDITIONAL_MIN__:
                self.container["conditional_cov", t] = coverage(lower[index], upper[index], y[index])
                self.container["conditional_width", t] = MPIW(lower[index], upper[index], y[index])
                self.container["conditional_perf", t] = self.container.perf_func(y[index], preds[index])

        # Check performance WITH conformalization
        if self.container.cp_mode():
            threshold = np.array([self.conditional_thresholds[taxonomy[i]] for i in range(len(taxonomy))])

            lower = lower - threshold
            upper = upper + threshold

            for t in sorted(self.conditional_thresholds.keys()):
                index = (taxonomy == t)
                if np.sum(index) > __CONDITIONAL_MIN__:
                    self.container["conditional_cp_cov", t] = coverage(lower[index], upper[index], y[index])
                    self.container["conditional_cp_width", t] = MPIW(lower[index], upper[index], y[index])
                    self.container["conditional_cp_perf", t] = self.container.perf_func(y[index], preds[index])

class RandomEnsembleConformalizer(EnsembleConformalizer):

    def train(self, x, y):
        total = 0
        for i in range(self.container.num()):
            super().train(x, y, suppress_verbosity = True)
            total += self.threshold
        self.threshold = total / self.container.num()

        if self.container.verbosity() > 1:
            print("Conformalized with threshold:", self.threshold)
            print()

    # NOT USED YET
    def train_conditional(self, x, y):
        super().train_conditional(x, y)
        total = self.conditional_thresholds
        for i in range(self.container.num() - 1):
            super().train_conditional(x, y, suppress_verbosity = True)
            total = {k: total.get(k, 0) + self.conditional_thresholds.get(k, 0) for k in set(total)}
        self.conditional_thresholds = {k: total.get(k, 0) / self.container.num() for k in set(total)}
