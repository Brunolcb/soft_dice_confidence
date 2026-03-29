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

    h = lambda x : ((2**np.arange(n)[::-1]) * (Pyi[x] >= 0.5)).sum()
    Y_pred = np.vectorize(h)(X)
    print(Y_pred[4])

    def compute_bounds(x):
        k = Y_pred[x].bit_count()

        if k == 0:
            return np.nan

        mu = np.where([yi_hat == '1' for yi_hat in np.binary_repr(Y_pred[x], width=n)], Pyi[x], 0).sum() / k
        s0 = np.where([yi_hat == '0' for yi_hat in np.binary_repr(Y_pred[x], width=n)], Pyi[x], 0).sum()

        r_b_U = b_U(k, mu, s0)
        r_b_L = b_L(k, mu, s0)

        return r_b_U, r_b_L
    results["bounds"] = Parallel(n_jobs=-1)(delayed(compute_bounds)(x) for x in X)

    def d_y_x(y, x):
        return dice_coeff(y, Y_pred[x]) * Py[x,y]
    def d(x):
        return np.vectorize(lambda y: d_y_x(y, x))(Y).sum()
    idc_fp = np.array(Parallel(n_jobs=-1)(delayed(d)(x) for x in X))  # true conditional risk
    results["IDC_FP"] = idc_fp

    R_x = 1 - idc_fp  # conditional risk
    results["R_x"] = R_x

    idc_fp_risks, idc_fp_coverages = rc_curve_(R_x, idc_fp)
    print("IDC_FP = ", auc(idc_fp_coverages, idc_fp_risks))
    curves["IDC_FP"] = (idc_fp_coverages, idc_fp_risks)

    def d_y_x_hat(y, x):
        y_probs = np.where([yi == '1' for yi in np.binary_repr(y, width=n)],
                           Pyi[x], 1 - Pyi[x])
        return dice_coeff(y, Y_pred[x]) * np.prod(y_probs)
    def d_hat(x):
        return np.vectorize(lambda y: d_y_x_hat(y, x))(Y).sum()
    idc_mar = np.array(Parallel(n_jobs=-1)(delayed(d_hat)(x) for x in X))  # approx. conditional risk
    results["IDC_MAR"] = idc_mar

    idc_mar_risks, idc_mar_coverages = rc_curve_(R_x, idc_mar)
    print("IDC_MAR = ", auc(idc_mar_coverages, idc_mar_risks))
    curves["IDC_MAR"] = (idc_mar_coverages, idc_mar_risks)

    aMSP_conf = np.vectorize(lambda x: aMSP(Y_pred[x], Pyi[x]))(X)
    results["aMSP"] = aMSP_conf
    risks, coverages = rc_curve_(R_x, aMSP_conf)
    print("Avg. MSP = ", auc(coverages, risks))
    curves["aMSP"] = (coverages, risks)

    sdc_conf = np.vectorize(lambda x: SDC(Y_pred[x], Pyi[x]))(X)
    results["SDC"] = sdc_conf
    sdc_risks, coverages = rc_curve_(R_x, sdc_conf)
    print("SDC = ", auc(coverages, sdc_risks))
    curves["SDC"] = (coverages, sdc_risks)

    with open(f"ideal_model_n_{n}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(f"ideal_model_n_{n}_rc_curve_small.pkl", "wb") as f:
        pickle.dump(curves, f)


def plot_varying_prevalence():
    def compute_metrics(loc):
        results = dict()
        aucs = dict()

        Py, Pyi, _ = get_posterior_and_marginals(loc)
        expected_prevalence = np.mean([sum(Py[x,y] * y.bit_count() for y in Y) for x in X]) / n
        aucs["expected_prevalence"] = expected_prevalence

        def h(x): return ((2**np.arange(n)[::-1]) * (Pyi[x] >= 0.5)).sum()
        Y_pred = np.vectorize(h)(X)

        def d_y_x(y, x):
            return dice_coeff(y, Y_pred[x]) * Py[x,y]
        def d(x):
            return np.vectorize(lambda y: d_y_x(y, x))(Y).sum()
        idc_fp = np.vectorize(d)(X)  # conditional risk
        results["IDC_FP"] = idc_fp
        
        R_x = 1 - idc_fp  # true conditional risk
        results["R_x"] = R_x

        idc_fp_risks, idc_fp_coverages = rc_curve_(R_x, idc_fp)
        aucs["IDC_FP"] = auc(idc_fp_coverages, idc_fp_risks)

        def d_y_x_hat(y, x):
            y_probs = np.where([yi == '1' for yi in np.binary_repr(y, width=n)],
                            Pyi[x], 1 - Pyi[x])
            return dice_coeff(y, Y_pred[x]) * np.prod(y_probs)
        def d_hat(x):
            return np.vectorize(lambda y: d_y_x_hat(y, x))(Y).sum()
        idc_mar = np.vectorize(d_hat)(X)  # approx. conditional risk
        results["IDC_MAR"] = idc_mar

        idc_mar_risks, idc_mar_coverages = rc_curve_(R_x, idc_mar)
        aucs["IDC_MAR"] = auc(idc_mar_coverages, idc_mar_risks)

        sdc_conf = np.vectorize(lambda x: SDC(Y_pred[x], Pyi[x]))(X)
        results["SDC"] = sdc_conf
        sdc_risks, sdc_coverages = rc_curve_(
            R_x, sdc_conf)
        aucs["SDC"] = auc(sdc_coverages, sdc_risks)

        aMSP_conf = np.vectorize(lambda x: aMSP(Y_pred[x], Pyi[x]))(X)
        results["aMSP"] = aMSP_conf
        msp_risks, msp_coverages = rc_curve_(
            R_x, aMSP_conf)
        aucs["aMSP"] = auc(msp_coverages, msp_risks)

        aucs["random"] = idc_fp_risks[-1]

        return results, aucs
    results = Parallel(n_jobs=-1)(delayed(compute_metrics)(loc) for loc in np.arange(-8, step=-0.5))
    resultss, aucss = zip(*results)

    with open(f"ideal_model_n_{n}_varying_prev_results.pkl", "wb") as f:
        pickle.dump(resultss, f)

    with open(f"ideal_model_n_{n}_varying_prev.pkl", "wb") as f:
        pickle.dump({
            "prevalences": [aucss[i]["expected_prevalence"] for i in range(len(resultss))],
            "sdc_aucs": [aucss[i]["SDC"] for i in range(len(aucss))],
            "msp_aucs": [aucss[i]["aMSP"] for i in range(len(aucss))],
            "idc_fp_aucs": [aucss[i]["IDC_FP"] for i in range(len(aucss))],
            "idc_mar_aucs": [aucss[i]["IDC_MAR"] for i in range(len(aucss))],
            "random_aucs": [aucss[i]["random"] for i in range(len(aucss))],
        }, f)


t = time()
plot_rc_curve_small()
plot_varying_prevalence()
print("Elapsed time:", time() - t)
