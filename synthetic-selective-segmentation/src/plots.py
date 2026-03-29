import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.metrics import auc

from .utils import *
from .risks import *


def plot_rc_curve(ax, r_hat, h, Py, loss, name, ideal_aurc=None, random_aurc=None):
    risks, coverages = rc_curve(-np.vectorize(r_hat)(X), h, Py, loss)
    aurc = auc(coverages, risks)

    if ideal_aurc is not None:
        aurc = (aurc - ideal_aurc)
        if random_aurc is not None:
            aurc = aurc / (random_aurc - ideal_aurc)

    ax.plot(coverages, risks, label=name + f"({aurc:.4f})")

def plot_xxx_rc(h, Py, hatPyi, loss, name, ax=None):
    if ax is None:
        ax = plt.gca()

    ## CONDITIONAL RISK
    r = get_true_conditional_risk(loss, Py, h)
    R_x = np.vectorize(r)(X)

    ## IDEAL
    risks, coverages = rc_curve(-R_x, h, Py, loss)
    ax.plot(coverages, risks, '--', c='black', label="Ideal")

    ideal_aurc = auc(coverages, risks)
    random_aurc = np.max(risks)

    ## MSP
    plot_rc_curve(ax, get_avg_msp_rhat(h, hatPyi), h, Py, loss, "1-AvgMSP", ideal_aurc, random_aurc)

    ## 01 RISK
    plot_rc_curve(ax, get_01_rhat(h, hatPyi), h, Py, loss, r"0-1 $\hat{r}$", ideal_aurc, random_aurc)

    ## Soft Volume Difference
    plot_rc_curve(ax, get_softvoldiff(h, hatPyi), h, Py, loss, r"VolDiff($p$, $\hat{y}$)", ideal_aurc, random_aurc)

    ## Quad volume difference
    plot_rc_curve(ax, get_quadvoldiff_rhat(h, hatPyi), h, Py, loss, r"QuadVol $\hat{r}$", ideal_aurc, random_aurc)

    ## SDC
    plot_rc_curve(ax, get_softdice_risk(h, hatPyi), h, Py, loss, r"1-SDC", ideal_aurc, random_aurc)

    # ylim = plt.ylim()
    # plt.ylim(0, ylim[1])
    ax.set_xlim(0, 1)
    ax.grid()
    ax.legend(title=r"$\hat{r}$ (NAURC)")
    ax.set_title(name)
    ax.set_ylabel("Risk")
    ax.set_xlabel("Coverage")

def plot_01_rc(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: 1-int(y == y_hat)

    plot_xxx_rc(h, Py, hatPyi, loss, '0-1 loss', ax=ax)

def plot_accuracy_rc(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: 1 - (int2img(y) == int2img(y_hat)).mean()

    plot_xxx_rc(h, Py, hatPyi, loss, 'Accuracy loss', ax=ax)

def plot_voldiff_rc(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: np.abs(int2img(y).sum(dtype=float) - int2img(y_hat).sum(dtype=float)) / 9

    plot_xxx_rc(h, Py, hatPyi, loss, 'Volume Difference loss', ax=ax)

def plot_quadvoldiff_rc(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: (int2img(y).sum(dtype=float) - int2img(y_hat).sum(dtype=float))**2 / 9

    plot_xxx_rc(h, Py, hatPyi, loss, 'Quadratic Volume Difference loss', ax=ax)

def plot_dice_rc(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: 1 - 2 * (int2img(y) * int2img(y_hat)).sum() / (int2img(y).sum() + int2img(y_hat).sum()+1e-3)

    plot_xxx_rc(h, Py, hatPyi, loss, 'Dice loss', ax=ax)

def plot_r_hat(ax, r_hat, R_x, name=""):
    R_hat_x = np.vectorize(r_hat)(X)
    spearman = sp.stats.spearmanr(R_x, R_hat_x)
    spearmanr = spearman.statistic
    # pvalue = spearman.pvalue
    ax.scatter(R_x, R_hat_x, s=2, label=name+f' (r={spearmanr:.4f})')

def plot_xxx_loss(h, Py, hatPyi, loss, rhat, name, ax=None):
    if ax is None:
        ax = plt.gca()

    ## CONDITIONAL RISK
    r = get_true_conditional_risk(loss, Py, h)
    R_x = np.vectorize(r)(X)

    R_h = R_x.mean()

    ## APPROXIMATE CONDITIONAL RISK
    if rhat is not None:
        plot_r_hat(ax, rhat, R_x, name=r"$\hat{r}$")

    ## MSP
    plot_r_hat(ax, get_avg_msp_rhat(h, hatPyi), R_x, name="1-AvgMSP")

    ## 01 RISK
    plot_r_hat(ax, get_01_rhat(h, hatPyi), R_x, name=r"0-1 $\hat{r}$")

    ## Soft Volume Difference
    plot_r_hat(ax, get_softvoldiff(h, hatPyi), R_x, name=r"VolDiff($p$, $\hat{y}$)")

    ## Quad volume difference
    plot_r_hat(ax, get_quadvoldiff_rhat(h, hatPyi), R_x, name=r"quadratic $\hat{r}$")

    ## SDC
    plot_r_hat(ax, get_softdice_risk(h, hatPyi), R_x, name='1-SDC')

    ax.set_title(name+f"\n$R(h)={R_h:.2f}$")
    ax.grid()
    ax.legend()
    ax.set_xlabel('$r(x)$')

def plot_01_loss(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: 1-int(all(y == y_hat))

    plot_xxx_loss(h, Py, hatPyi, loss, None, '0-1 loss', ax=ax)

def plot_accuracy_loss(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: 1 - (int2img(y) == int2img(y_hat)).mean()

    plot_xxx_loss(h, Py, hatPyi, loss, None, 'Accuracy loss', ax=ax)

def plot_voldiff_loss(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: np.abs(int2img(y).sum(dtype=float) - int2img(y_hat).sum(dtype=float)) / 9

    plot_xxx_loss(h, Py, hatPyi, loss, None, 'Volume Difference loss', ax=ax)

def plot_quadvoldiff_loss(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: (int2img(y).sum(dtype=float) - int2img(y_hat).sum(dtype=float))**2 / 9

    plot_xxx_loss(h, Py, hatPyi, loss, None, 'Quadratic Volume Difference loss', ax=ax)

def plot_dice_loss(h, Py, hatPyi, ax=None):
    loss = lambda y, y_hat: 1 - 2 * (int2img(y) * int2img(y_hat)).sum() / (int2img(y).sum() + int2img(y_hat).sum()+1e-3)

    plot_xxx_loss(h, Py, hatPyi, loss, None, 'Dice loss', ax=ax)

def plot_noise_xxx_loss(loss, get_h, Py, Pyi, name, noise_range=[0.0, 0.5, 1.0, 2.0, 5.0], ax=None):
    if ax is None:
        ax = plt.gca()

    get_r_hats = {
        r'1-AvgMSP': get_avg_msp_rhat,
        r'0-1 $\hat{r}$': get_01_rhat,
        r'VolDiff': get_softvoldiff,
        r'SVE $\hat{r}$': get_quadvoldiff_rhat,
        r'1-SDC': get_softdice_risk,
    }
    naurcs = {
        r'1-AvgMSP': list(),
        r'0-1 $\hat{r}$': list(),
        r'VolDiff': list(),
        r'SVE $\hat{r}$': list(),
        r'1-SDC': list(),
    }

    for s in noise_range:
        hatPyi, hatPy = get_hatPyi(Pyi, scale=s)
        h = get_h(hatPy)

        r = get_true_conditional_risk(loss, Py, h)
        R = np.vectorize(r)(X)
        risks, coverages = rc_curve(-R, h, Py, loss)

        ideal_aurc = auc(coverages, risks)
        random_aurc = np.max(risks)

        for r_name in naurcs.keys():
            r_hat = get_r_hats[r_name](h, hatPyi)

            risks, coverages = rc_curve(-np.vectorize(r_hat)(X), h, Py, loss)
            naurc = (auc(coverages, risks) - ideal_aurc) / (random_aurc - ideal_aurc)
            naurcs[r_name].append(naurc)

    for r_name in naurcs.keys():
        ax.plot(noise_range, naurcs[r_name], '--o', label=r_name)

    ax.legend()
    ax.set_title(name+" Loss")
    ax.set_xlabel("Noise range")
    ax.set_ylabel("NAURC")

def plot_noise_01_loss(get_h, Py, Pyi, noise_range=[0.0, 0.5, 1.0, 2.0, 5.0], ax=None):
    loss = lambda y, y_hat: 1 - (int2img(y) == int2img(y_hat)).mean()

    plot_noise_xxx_loss(loss, get_h, Py, Pyi, "01", noise_range=noise_range, ax=ax)

def plot_noise_accuracy_loss(get_h, Py, Pyi, noise_range=[0.0, 0.5, 1.0, 2.0, 5.0], ax=None):
    loss = lambda y, y_hat: 1-int(y == y_hat)

    plot_noise_xxx_loss(loss, get_h, Py, Pyi, "Acc", noise_range=noise_range, ax=ax)

def plot_noise_voldiff_loss(get_h, Py, Pyi, noise_range=[0.0, 0.5, 1.0, 2.0, 5.0], ax=None):
    loss = lambda y, y_hat: np.abs(int2img(y).sum(dtype=float) - int2img(y_hat).sum(dtype=float)) / 9

    plot_noise_xxx_loss(loss, get_h, Py, Pyi, "AVE", noise_range=noise_range, ax=ax)

def plot_noise_quadvoldiff_loss(get_h, Py, Pyi, noise_range=[0.0, 0.5, 1.0, 2.0, 5.0], ax=None):
    loss = lambda y, y_hat: (int2img(y).sum(dtype=float) - int2img(y_hat).sum(dtype=float))**2 / 9

    plot_noise_xxx_loss(loss, get_h, Py, Pyi, "SVE", noise_range=noise_range, ax=ax)

def plot_noise_dice_loss(get_h, Py, Pyi, noise_range=[0.0, 0.5, 1.0, 2.0, 5.0], ax=None):
    loss = lambda y, y_hat: 1 - 2 * (int2img(y) * int2img(y_hat)).sum() / (int2img(y).sum() + int2img(y_hat).sum()+1e-3)

    plot_noise_xxx_loss(loss, get_h, Py, Pyi, "DSC", noise_range=noise_range, ax=ax)
