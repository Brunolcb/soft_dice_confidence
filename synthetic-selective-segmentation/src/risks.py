import numpy as np
import scipy as sp

from .utils import *


def get_true_conditional_risk(loss, Py, h, n=9):
    def r(x):
        y_hat = h(x)
        R_Y_x = np.vectorize(lambda y: loss(y, y_hat))(np.arange(2**n))
        return Py[x] @ R_Y_x
    
    return r

def get_01_rhat(h, hatPyi):
    return lambda x: 1 - np.where(int2img(h(x)), hatPyi[x], 1 - hatPyi[x]).prod()

def get_avg_msp_rhat(h, hatPyi):
    return lambda x: 1 - np.where(int2img(h(x)), hatPyi[x], 1 - hatPyi[x]).mean()

def get_softvoldiff(h, hatPyi):
    return lambda x: np.abs(hatPyi[x].sum(dtype=float) - int2img(h(x)).sum(dtype=float)) / 9

def get_quadvoldiff_rhat(h, hatPyi):
    def r_hat(x):
        # TODO: replace with O(n) version
        y_hat = int2img(h(x)).flatten()
        p_hat = hatPyi[x].flatten()
        
        return (np.sum(y_hat - p_hat)**2 + np.sum(p_hat - p_hat**2)) / 9

    return r_hat

def get_softdice_risk(h, hatPyi):
    return lambda x: 1 - 2 * np.sum(hatPyi[x] * int2img(h(x))) / (hatPyi[x].sum() + int2img(h(x)).sum())

def expected_risk(h, Px, Py, loss):
    r = get_true_conditional_risk(loss, Py, h)

    R_x = np.vectorize(r)(X)

    return R_x @ Px[X]

def rc_curve(confidence, h, Py, loss, n=9):
    r = get_true_conditional_risk(loss, Py, h)

    R_x = np.vectorize(r)(np.arange(2**n))

    return rc_curve_(R_x, confidence)

def rc_curve_(errors, confidence):
    errors = np.copy(errors) / len(errors)

    desc_sort_indices = confidence.argsort()[::-1]

    errors = errors[desc_sort_indices]

    coverages = np.cumsum(np.ones_like(errors) / len(errors))
    risks = np.cumsum(errors) / coverages

    return risks, coverages

def get_hatPyi(Pyi, scale=1.0):
    # pixel-level probability is predicted with a gaussian noise at the logits
    def hatpyi(x):
        z = sp.special.logit(Pyi[x])
        zhat = np.random.RandomState(x).normal(z, scale=scale)
        return sp.special.expit(zhat)

    hatPyi = [hatpyi(x) for x in np.arange(2**9)]
    hatPy = [np.array([np.where(int2img(y), pyi, 1 - pyi).prod() for y in np.arange(2**9)]) for pyi in hatPyi]

    return hatPyi, hatPy
