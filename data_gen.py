import numpy as np
import sklearn

def data_gen():
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    return X, y


