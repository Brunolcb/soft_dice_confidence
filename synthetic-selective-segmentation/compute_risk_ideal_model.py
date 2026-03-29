import pickle

from joblib import Parallel, delayed
from time import time

import numpy as np

from src import *

# Dice loss function on binary numbers
dice_loss = lambda y, y_hat: 1 - 2 * bin(y & y_hat).count("1") / (bin(y).count("1") + bin(y_hat).count("1") + 1e-3)

# Confidence Estimators
def SDC(y_hat, hatPyi_x):
    # y_hat_ = int2img(y_hat, n=n).flatten()
    # return 2 * (y_hat_ * hatPyi).sum() / (np.sum(y_hat_) + hatPyi.sum())
    n = hatPyi_x.shape[0]
    tp = np.where([yi_hat == '1' for yi_hat in np.binary_repr(y_hat, width=n)], hatPyi_x, 0).sum()
    return 2 * tp / (bin(y_hat).count("1") + hatPyi_x.sum())

def aMSP(y_hat, hatPyi_x):
    n = hatPyi_x.shape[0]
    return np.where([yi_hat == '1' for yi_hat in np.binary_repr(y_hat, width=n)], hatPyi_x, 1-hatPyi_x).mean()

N = 1000  # number of images

# number of pixels
def compute_for_n(n):
    # marginal probabilities
    Pyi = np.array([sp.special.expit(np.random.RandomState(2**n + x - 1).normal(0, scale=5, size=n)) for x in range(N)])

    h = lambda x: ((2**np.arange(n)[::-1]) * (Pyi[x] >= 0.5)).sum()

    Y_pred = np.vectorize(h)(np.arange(N))

    # image-level output probability derived from pixel-level probabilities
    def get_Py_x(y, x, Pyi=Pyi):
        y_probs = np.where([yi == '1' for yi in np.binary_repr(y, width=n)], Pyi[x], 1 - Pyi[x])
        return np.prod(y_probs)

    def r_y_x(y, x):
        y_pred = Y_pred[x]
        return dice_loss(y, y_pred) * get_Py_x(y, x)

    def r(x):
        return np.vectorize(lambda y: r_y_x(y, x))(np.arange(2**n)).sum()

    R_x = np.vectorize(r)(np.arange(N))  # conditional risk

    return {
        "n": n,
        "Pyi": Pyi,
        "R_x": R_x,
    }

t0 = time()
results = Parallel(n_jobs=-1)(delayed(compute_for_n)(n) for n in np.arange(1, 20, 2))

print("Elapsed time:", time() - t0)

with open("results_ideal.pkl", "wb") as f:
    pickle.dump(results, f)
