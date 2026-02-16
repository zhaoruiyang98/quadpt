import numpy as np
from ._fastgl import leggauss as _leggauss  # type: ignore


def leggauss(deg: int):
    w = np.zeros(deg, dtype=np.float64)
    x = np.zeros(deg, dtype=np.float64)
    _leggauss(deg, x, w)
    return x, w
