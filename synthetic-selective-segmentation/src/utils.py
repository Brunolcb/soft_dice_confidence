import numpy as np


X = np.arange(1, 2**9)

def int2img(value, n=9):
    w = np.sqrt(n).astype(int)
    return (value & (1 << np.arange(n)) > 0).astype('uint8').reshape(w,w)

def softmax_prob(x, h, hatPyi):
    return np.where(int2img(h(x)), hatPyi[x], 1 - hatPyi[x])

def get_expected(Px, Py, h, metric):
    expected = 0
    for x in X:
        y_hat = h(x)
        for y in range(2**9):
            expected += Px[x] * Py[x][y] * metric(y, y_hat)
    return expected

# def rc_curve(confidence, h, Px, Py, loss):
#     r = get_true_conditional_risk(loss, Py, h)

#     R_x = np.vectorize(r)(X)

#     expected_risk_X = R_x * Px[X]

#     error = np.array(error).reshape(-1)
#     confidence = np.array(confidence).reshape(-1)
#     n = len(error)
#     assert len(confidence) == n
#     desc_sort_indices = confidence.argsort()[::-1]
#     error = error[desc_sort_indices]
#     confidence = confidence[desc_sort_indices]
#     idx = np.r_[np.where(np.diff(confidence))[0], n-1]
#     thresholds = confidence[idx]
#     coverages = (1 + idx)/n
#     risks = np.cumsum(error)[idx]/n
#     risks /= coverages

#     coverages = np.insert(coverages, 0, 0)
#     risks = np.insert(risks, 0, risks[0])
#     thresholds = np.insert(thresholds, 0, 0)

#     return coverages, risks, thresholds
