import pickle
from time import time
import numpy as np
from joblib import Parallel, delayed
from src import *

from utils import *


def plot_rc_curve_small():
    results = dict()
    curves = dict()

    loc = -3.698  # hand-tuned such that prevalence \approx 0.25 with normalized P(y) func

    Py, Pyi, _ = get_posterior_and_marginals(loc)

    expected_prevalence = np.mean([np.sum([Py[x,y] * y.bit_count() for y in Y]) for x in X]) / n
    print("Expected prevalence = ", expected_prevalence)

    # pixel-level probability is predicted with a gaussian noise at the logits
    def hatpyi(x, scale=2.0):
        z = sp.special.logit(Pyi[x])
        zhat = np.random.RandomState(x+33).normal(z, scale=scale)
        return sp.special.expit(zhat)
    hatPyi = np.vstack([hatpyi(x) for x in X])

    h_hat = lambda x: ((2**np.arange(n)[::-1]) * (hatPyi[x] >= 0.5)).sum()
    Y_hat = np.vectorize(h_hat)(X)

    def d_y_x(y, x):
        return dice_coeff(y, Y_hat[x]) * Py[x,y]
    def d(x):
        return np.vectorize(lambda y: d_y_x(y, x))(Y).sum()
    idc_fp = np.vectorize(d)(X)  # conditional risk
    results["IDC_FP"] = idc_fp
    
    R_x = 1 - idc_fp  # conditional risk
    results["R_x"] = R_x

    idc_fp_risks, idc_fp_coverages = rc_curve_(R_x, idc_fp)
    print("IDC_FP = ", auc(idc_fp_coverages, idc_fp_risks))
    curves["IDC_FP"] = (idc_fp_coverages, idc_fp_risks)

    risks, coverages = rc_curve_(R_x, aMSP_conf := np.vectorize(lambda x: aMSP(Y_hat[x], hatPyi[x]))(X))
    results["aMSP"] = aMSP_conf
    print("Avg. MSP = ", auc(coverages, risks))
    curves["aMSP"] = (coverages, risks)

    sdc_risks, coverages = rc_curve_(R_x, sdc_conf := np.vectorize(lambda x: SDC(Y_hat[x], hatPyi[x]))(X))
    results["SDC"] = sdc_conf
    print("SDC = ", auc(coverages, sdc_risks))
    curves["SDC"] = (coverages, sdc_risks)

    # using p
    def d_y_x_hat(y, x):
        y_probs = np.where([yi == '1' for yi in np.binary_repr(y, width=n)],
                           Pyi[x], 1 - Pyi[x])
        return dice_coeff(y, Y_hat[x]) * np.prod(y_probs)
    def d_hat(x):
        return np.vectorize(lambda y: d_y_x_hat(y, x))(Y).sum()
    idc_mar = np.array(Parallel(n_jobs=-1)(delayed(d_hat)(x) for x in X))  # approx. conditional risk
    results["IDC_MAR"] = idc_mar

    idc_mar_risks, idc_mar_coverages = rc_curve_(R_x, idc_mar)
    curves["IDC_MAR"] = (idc_mar_coverages, idc_mar_risks)

    # using \hat{p}
    def hat_d_y_x(y, x):
        hat_py_x = np.prod(np.where([yi == '1' for yi in np.binary_repr(y, width=n)], hatPyi[x], 1 - hatPyi[x]))
        return dice_coeff(y, Y_hat[x]) * hat_py_x
    def hat_d(x):
        return np.vectorize(lambda y: hat_d_y_x(y, x))(Y).sum()
    idc_hat = np.vectorize(hat_d)(X)  # predicted conditional risk
    results["IDC_hat"] = idc_hat

    idc_hat_risks, idc_hat_coverages = rc_curve_(R_x, idc_hat)
    curves["IDC_hat"] = (idc_hat_coverages, idc_hat_risks)

    with open(f"perturbed_model_n_{n}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(f"perturbed_model_n_{n}_rc_curve_small.pkl", "wb") as f:
        pickle.dump(curves, f)


def plot_varying_perturbation():
    loc = -3.698  # hand-tuned such that prevalence \approx 0.25 with normalized P(y) func
    Py, Pyi, _ = get_posterior_and_marginals(loc)

    # pixel-level probability is predicted with a gaussian noise at the logits
    def hatpyi(x, scale=2.0):
        z = sp.special.logit(Pyi[x])
        zhat = np.random.RandomState(x+33).normal(z, scale=scale)
        return sp.special.expit(zhat)

    def compute_metrics(scale):
        results = dict()
        aucs = dict()

        hatPyi = np.vstack([hatpyi(x, scale=scale) for x in X])

        h_hat = lambda x: ((2**np.arange(n)[::-1]) * (hatPyi[x] >= 0.5)).sum()
        Y_hat = np.vectorize(h_hat)(X)

        def d_y_x(y, x):
            return dice_coeff(y, Y_hat[x]) * Py[x,y]
        def d(x):
            return np.vectorize(lambda y: d_y_x(y, x))(Y).sum()
        idc_fp = np.vectorize(d)(X)  # conditional risk
        results["IDC_FP"] = idc_fp

        R_x = 1 - idc_fp  # conditional risk
        results["R_x"] = R_x

        sdc_risks, sdc_coverages = rc_curve_(R_x, sdc_conf := np.vectorize(lambda x: SDC(Y_hat[x], hatPyi[x]))(X))
        results["SDC"] = sdc_conf
        aucs["SDC"] = auc(sdc_coverages, sdc_risks)

        msp_risks, msp_coverages = rc_curve_(R_x, msp_conf := np.vectorize(lambda x: aMSP(Y_hat[x], hatPyi[x]))(X))
        results["aMSP"] = msp_conf
        aucs["aMSP"] = auc(msp_coverages, msp_risks)

        idc_fp_risks, idc_fp_coverages = rc_curve_(R_x, idc_fp)
        aucs["IDC_FP"] = auc(idc_fp_coverages, idc_fp_risks)

        aucs["Random"] = idc_fp_risks[-1]

        # using p
        def d_y_x_hat(y, x):
            y_probs = np.where([yi == '1' for yi in np.binary_repr(y, width=n)],
                            Pyi[x], 1 - Pyi[x])
            return dice_coeff(y, Y_hat[x]) * np.prod(y_probs)
        def d_hat(x):
            return np.vectorize(lambda y: d_y_x_hat(y, x))(Y).sum()
        idc_mar = np.array(Parallel(n_jobs=-1)(delayed(d_hat)(x) for x in X))  # approx. conditional risk
        results["IDC_MAR"] = idc_mar

        idc_mar_risks, idc_mar_coverages = rc_curve_(R_x, idc_mar)
        aucs["IDC_MAR"] = auc(idc_mar_coverages, idc_mar_risks)

        def hat_d_y_x(y, x):
            hat_py_x = np.prod(np.where([yi == '1' for yi in np.binary_repr(y, width=n)], hatPyi[x], 1 - hatPyi[x]))
            return dice_coeff(y, Y_hat[x]) * hat_py_x
        def hat_d(x):
            return np.vectorize(lambda y: hat_d_y_x(y, x))(Y).sum()
        idc_hat = np.vectorize(hat_d)(X)  # conditional risk
        results["IDC_hat"] = idc_hat

        idc_hat_risks, idc_hat_coverages = rc_curve_(R_x, idc_hat)
        aucs["IDC_hat"] = auc(idc_hat_coverages, idc_hat_risks)

        aucs["risk"] = idc_fp_risks[-1]

        return results, aucs

    results = Parallel(n_jobs=-1)(delayed(compute_metrics)(loc)
                                  for loc in np.arange(5, step=0.5))
    resultss, aucss = zip(*results)

    with open(f"perturbed_model_n_{n}_varying_perturbation_results.pkl", "wb") as f:
        pickle.dump(resultss, f)

    with open(f"perturbed_model_n_{n}_varying_perturbation.pkl", "wb") as f:
        pickle.dump({
            "risks": [a["risk"] for a in aucss],
            "sdc_aucs": [a["SDC"] for a in aucss],
            "msp_aucs": [a["aMSP"] for a in aucss],
            "idc_fp_aucs": [a["IDC_FP"] for a in aucss],
            "idc_hat_aucs": [a["IDC_hat"] for a in aucss],
            "random_aucs": [a["Random"] for a in aucss],
        }, f)


t = time()
plot_rc_curve_small()
plot_varying_perturbation()
print("Elapsed time:", time() - t)
