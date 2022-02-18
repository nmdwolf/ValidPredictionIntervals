from Code.CP import *

import numpy as np

from sklearn.metrics import r2_score
from torch.nn import MSELoss

class DataObject:

    def __init__(self, dataset, method, seed, alpha = 0.1, test_frac = 0.05, \
    minimal_cov = 0.6, perf_func = r2_score, taxonomy_func = None, \
    disconformity_measure = DISTscore, verbosity = 0, cp_mode = True, extra = None):

        self.__alpha = alpha
        self.__cp_mode = cp_mode
        self.__dataset = dataset
        self.__extra = extra if extra else dict()
        self.score = disconformity_measure

        self.__method = method
        self.__minimal_cov = minimal_cov
        self.perf_func = perf_func
        self.__seed = seed
        self.__taxonomy_func = taxonomy_func
        self.__test_frac = test_frac
        self.__verbosity = verbosity

        self.__cov = -999
        self.__width = -999
        self.__perf = -999

        self.__cp_cov = -999
        self.__cp_width = -999
        self.__cp_perf = -999

        if self.__taxonomy_func:
            self.__conditional_cov = dict()
            self.__conditional_width = dict()
            self.__conditional_perf = dict()
            for i in range(1, self.__taxonomy_func.split_count()+1):
                self.__conditional_cov[i] = -999
                self.__conditional_width[i] = -999
                self.__conditional_perf[i] = -999
            self.__conditional_cp_cov = dict()
            self.__conditional_cp_width = dict()
            self.__conditional_cp_perf = dict()
            for i in range(1, self.__taxonomy_func.split_count()+1):
                self.__conditional_cp_cov[i] = -999
                self.__conditional_cp_width[i] = -999
                self.__conditional_cp_perf[i] = -999

    def __getitem__(self, type):
        if type == "cov":
            return self.__cov
        elif type == "width":
            return self.__width
        elif type == "perf":
            return self.__perf
        elif type == "cp_cov":
            return self.__cp_cov
        elif type == "cp_width":
            return self.__cp_width
        elif type == "cp_perf":
            return self.__cp_perf
        elif type == "conditional_cov":
            return self.__conditional_cov
        elif type == "conditional_width":
            return self.__conditional_width
        elif type == "conditional_perf":
            return self.__conditional_perf
        elif type == "conditional_cp_cov":
            return self.__conditional_cp_cov
        elif type == "conditional_cp_width":
            return self.__conditional_cp_width
        elif type == "conditional_cp_perf":
            return self.__conditional_cp_perf
        elif type in self.__extra.keys():
            return self.__extra[type]

    def __setitem__(self, type, value):
        if type == "cov":
            self.__cov = value
        elif type == "width":
            self.__width = value
        elif type == "perf":
            self.__perf = value
        elif type == "cp_cov":
            self.__cp_cov = value
        elif type == "cp_width":
            self.__cp_width = value
        elif type == "cp_perf":
            self.__cp_perf = value
        elif type[0] == "conditional_cov":
            self.__conditional_cov[type[1]] = value
        elif type[0] == "conditional_width":
            self.__conditional_width[type[1]] = value
        elif type[0] == "conditional_perf":
            self.__conditional_perf[type[1]] = value
        elif type[0] == "conditional_cp_cov":
            self.__conditional_cp_cov[type[1]] = value
        elif type[0] == "conditional_cp_width":
            self.__conditional_cp_width[type[1]] = value
        elif type[0] == "conditional_cp_perf":
            self.__conditional_cp_perf[type[1]] = value

    def alpha(self):
        return self.__alpha

    def cp_mode(self):
        return self.__cp_mode

    def dataset(self):
        return self.__dataset

    def method(self):
        return self.__method

    def minimal_cov(self):
        return self.__minimal_cov

    def seed(self):
        return self.__seed

    def taxonomy_func(self):
        return self.__taxonomy_func

    def test_frac(self):
        return self.__test_frac

    def verbosity(self):
        return self.__verbosity

    def generateFile(self, CP = False):

        if CP:
            result = str(self.__cp_cov) + "\t" + str(self.__cp_width) + "\t" + str(self.__cp_perf) + "\n"
            conditional_result = ""
            if self.__taxonomy_func:
                for i in range(self.__taxonomy_func.split_count()):
                    conditional_result += str(self.__conditional_cp_cov[i+1]) + "\t" + str(self.__conditional_cp_width[i+1]) + "\t" + str(self.__conditional_cp_perf[i+1]) + "\n"
        else:
            result = str(self.__cov) + "\t" + str(self.__width) + "\t" + str(self.__perf) + "\n"
            conditional_result = ""
            if self.__taxonomy_func:
                for i in range(self.__taxonomy_func.split_count()):
                    conditional_result += str(self.__conditional_cov[i+1]) + "\t" + str(self.__conditional_width[i+1]) + "\t" + str(self.__conditional_perf[i+1]) + "\n"

        return result, conditional_result

    def export(self, folder, with_config = True, skip = True, extra = None):

        result, conditional_result = self.generateFile()
        cp_result, conditional_cp_result = self.generateFile(CP = True)

        if self.__cov != -999 and self.__width != -999 and self.__perf != -999:
            with open(folder + self.__dataset + "/results/" + self.__method + ("-FULL" if not self.__cp_mode else "") + "-results-seed" + str(self.__seed) + ".txt", "w") as file:
                file.write(result)

        if self.__cp_cov != -999 and self.__cp_width != -999 and self.__cp_perf != -999:
            with open(folder + self.__dataset + "/results/" + self.__method + "-CP-results-seed" + str(self.__seed) + ".txt", "w") as file:
                file.write(cp_result)

        if self.__taxonomy_func:
            if self.__cov != -999 and self.__width != -999 and self.__perf != -999:
                with open(folder + self.__dataset + "/results/" + self.__method + ("-FULL" if not self.__cp_mode else "") + "-conditional-results-seed" + str(self.__seed) + ".txt", "w") as file:
                    file.write(conditional_result)
            if self.__cp_cov != -999 and self.__cp_width != -999 and self.__cp_perf != -999:
                with open(folder + self.__dataset + "/results/" + self.__method + "-CP-conditional-results-seed" + str(self.__seed) + ".txt", "w") as file:
                    file.write(conditional_cp_result)

        if with_config:
            with open(folder + self.__dataset + "/results/" + self.__method + "-config.txt", "w") as file:
            # with open(folder + self.__dataset + "-" + self.__method + "-config-seed" + str(self.__seed) + ".txt", "w") as file:
                file.write(self.dump())
                for k in self.__extra.keys():
                    file.write(str(k) + ":\t" + str(self.__extra[k]) + "\n")

    def dump(self):
        return "alpha:\t"+str(self.__alpha)+"\n" + \
        "dataset:\t"+self.__dataset+"\n" + \
        "seed:\t"+str(self.__seed)+"\n" + \
        "method:\t"+self.__method+"\n" + \
        "test_frac:\t"+str(self.__test_frac)+"\n" + \
        "performance_measure:\t"+str(self.perf_func)+"\n" + \
        "disconformity_measure:\t"+str(self.score)+"\n" + \
        "taxonomy_func:\t"+str(self.__taxonomy_func)+"\n" + \
        "verbosity:\t"+str(self.__verbosity)+"\n"

class NNDataObject(DataObject):

    def __init__(self, dataset, method, seed, alpha = 0.1, batch = 64, dim = 64, \
    val_length = 1, drop = 0.1, l = 1e-6, epochs = 100, learning_rate = 5e-4, num = 1, \
    test_frac = 0.05, loss_func = MSELoss(), perf_func = r2_score, taxonomy_func = None, verbosity = 0, cp_mode = True, extra = None):

        super().__init__(dataset, method, seed, alpha, test_frac, perf_func = perf_func, taxonomy_func = taxonomy_func, disconformity_measure = DISTscore, verbosity = verbosity, cp_mode = cp_mode, extra = extra)
        self.__batch = batch
        self.__dim = dim
        self.__drop = drop
        self.__epochs = epochs
        self.__l = l
        self.__learning_rate = learning_rate
        self.loss_func = loss_func
        self.__num = num
        self.__val_length = val_length

    def batch(self):
        return self.__batch

    def dim(self):
        return self.__dim

    def drop(self):
        return self.__drop

    def epochs(self):
        return self.__epochs

    def l(self):
        return self.__l

    def learning_rate(self):
        return self.__learning_rate

    def num(self):
        return self.__num

    def val_length(self):
        return self.__val_length

    def dump(self):
        return super().dump() + \
        "batch:\t" + str(self.__batch) + "\n" + \
        "dim:\t" + str(self.__dim) + "\n" + \
        "drop:\t" + str(self.__drop) + "\n" + \
        "epochs:\t" + str(self.__epochs) + "\n" + \
        "l2:\t" + str(self.__l) + "\n" + \
        "learning_rate:\t" + str(self.__learning_rate) + "\n" + \
        "val_length:\t" + str(self.__val_length) + "\n" + \
        "ensemble_size:\t" + str(self.__num) + "\n" + \
        "loss_func:\t" + str(self.loss_func) + "\n"

class ForestDataObject(DataObject):

    def __init__(self, dataset, method, seed, alpha = 0.1, test_frac = 0.05, \
    trees = 100, perf_func = r2_score, taxonomy_func = None, verbosity = 0, cp_mode = True, extra = None):

        super().__init__(dataset, method, seed, alpha, test_frac, perf_func = perf_func, taxonomy_func = taxonomy_func, disconformity_measure = DISTscore, verbosity = verbosity, cp_mode = cp_mode, extra = None)
        self.__trees = trees

    def trees(self):
        return self.__trees

    def dump(self):
        return super().dump() + \
        "trees:\t" + str(self.__trees) + "\n"
