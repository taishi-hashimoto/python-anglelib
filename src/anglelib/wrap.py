"A function to wrap values within the specified range."
import numpy as np


def wrap(x, lo=-np.pi, hi=np.pi):
    "Wrap x in [lo, hi)."
    is_scalar = np.isscalar(x)
    y = np.atleast_1d(x)
    while np.any(y < lo):
        y[y < lo] += hi - lo
    while np.any(y >= hi):
        y[y >= hi] -= hi - lo
    if is_scalar:
        return y[0]
    else:
        return y
