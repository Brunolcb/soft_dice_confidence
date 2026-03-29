import numpy as np
import scipy as sp

# number of pixels
n = 10

N = 50000
X = np.arange(N)
Y0 = np.arange(2**n)
Y = np.arange(1, 2**n)  # we're excluding y=0!!!

# image-level output probability derived from pixel-level probabilities
def get_Py_x(y, x, Qyi):
    if y == 0:
        return 0
    else:
        y_probs = np.where([yi == '1' for yi in np.binary_repr(y, width=n)], Qyi[x], 1 - Qyi[x])
        empty_img_prob = np.prod(1 - Qyi[x])
        return np.prod(y_probs) / (1 - empty_img_prob)

def get_posterior_and_marginals(loc):
    # marginal probabilities from conditionally-independent model
    Qyi = np.array([sp.special.expit(np.random.RandomState(x).normal(loc, scale=5, size=n)) for x in X])

    # normalization to enforce P[Y=0] = 0
    Py = np.array([[get_Py_x(y, x, Qyi=Qyi) for y in np.arange(np.max(Y) + 1)] for x in X])
    Pyi = np.array([qyi_x / (1 - np.prod(1 - qyi_x)) for qyi_x in Qyi])

    assert np.isclose(Py.sum(axis=1), 1).all()  # Maybe this is the problem?

    return Py, Pyi, Qyi

# Dice loss function on binary numbers
def dice_coeff(y, y_hat):
    if y == 0 or y_hat == 0:
        return 0.0
    else:
        return 2 * bin(y & y_hat).count("1") / (bin(y).count("1") + bin(y_hat).count("1"))

def SDC(y_hat, hatPyi):
    tp = np.where([yi_hat == '1' for yi_hat in np.binary_repr(y_hat, width=n)], hatPyi, 0).sum()
    return 2 * tp / (bin(y_hat).count("1") + hatPyi.sum())


def aMSP(y_hat, hatPyi):
    return np.where([yi_hat == '1' for yi_hat in np.binary_repr(y_hat, width=n)], hatPyi, 1-hatPyi).mean()


def b_U(k, mu, lam, tol=1e-12):
    """
    Parameters:
        max_terms (int): Maximum number of terms to sum in the series.
        tol (float): Convergence tolerance.
    """
    numerator_constant = np.float128(k + k * mu + lam)  # Precompute numerator constant
    denominator_base = np.float128(k + k * mu)  # Precompute denominator base
    e_neg_lam = np.exp(-lam, dtype=np.float128)  # Precompute e^(-lambda)

    b_u_value = 0.0

    i = 0
    lam_to_the_i = lam ** 0
    cum_prob = 0
    poisson_term = e_neg_lam  # Poisson probability
    while True:
        fraction = numerator_constant / (denominator_base + i)  # Fraction term
        term = poisson_term * fraction  # Current term

        b_u_value += term
        cum_prob += poisson_term

        # Check for convergence
        # if term < tol:
        #     break
        if (cum_prob > 0.99) and (term < tol):
            break

        i += 1
        lam_to_the_i = lam_to_the_i * lam
        poisson_term = poisson_term * (lam / i)

    return b_u_value

def b_L(k, mu, lam):
    return (k + k*mu + lam) / (k + 1 + (k - 1)*mu + lam)
