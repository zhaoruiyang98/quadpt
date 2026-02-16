import itertools
import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import spherical_jn, loggamma


def itjv(v, x, extradps=0):
    """Integration of Bessel Jv function from 0 to x."""
    v = np.asarray(v)
    x = np.asarray(x)
    v, x = np.broadcast_arrays(v, x)

    if np.any(v < -1):
        raise ValueError("Order nu must be >= -1")

    special_case = v == -1
    w = np.where(special_case, -1.0, 1.0)
    v = np.where(special_case, 1.0, v)

    def hypval_scalar(v_, x_):
        # looks like results are all floats...
        return float(mp.hyp1f2(0.5 * (1 + v_), 0.5 * (3 + v_), 1 + v_, -x_ * x_ / 4))

    with mp.extradps(extradps):
        hypval = np.vectorize(hypval_scalar, otypes=[float])(v, x)
    logval = -v * np.log(2) + (1 + v) * np.log(x) + np.log(hypval) - loggamma(2 + v)
    return w * np.exp(logval)


def spherical_jn_zeros(n):
    """Generate zeros of spherical Bessel function j_n(x) for n >= 0. (n is int)"""
    if np.floor(n) != n:
        raise ValueError("Arguments must be integers.")
    if n == 0:
        for m in itertools.count(1):
            yield m * np.pi
        return
    prev_zeros = spherical_jn_zeros(n - 1)
    left = next(prev_zeros)
    for right in prev_zeros:
        # j_{n+1}'s zeros are located between j_n's zeros
        yield brentq(lambda x, n=n: spherical_jn(n, x), left, right)
        left = right


def solve_linear_recurrence(P, R, a, b, Ya, Yb=[]):
    """Algorithm from J. Oliver's paper: https://doi.org/10.1007/BF02166688 .

    Solves the following linear recurrence equation:
    P[0, i] * Y[i + n] + ... + P[n-1, i] * Y[i+1] + P[n, i] * Y[i]  = R[i]

    Parameters
    ----------
    P : callable function
        ufunc w.r.t. argument i, P(n, i)
    R : callable function
        ufunc w.r.t. argument i, R(i)
    a : int
        index of the first element of Ya
    b : int
        index of the last element of Yb
    Ya : array_like
        boundary conditions, Ya = Y[a], Y[a+1], ..., Y[a+n-m-1]
    Yb : array_like, by default empty
        boundary conditions, Yb = Y[b-m+1], Y[b-m+2], ..., Y[b]
        if not empty, it solves a boundary value problem instead of an initial value problem.

    Examples
    --------
    >>> P = lambda n, i: {0: -1, 1: 1, 2: 1}[n] * np.ones_like(i)
    >>> R = lambda i: np.zeros_like(i)
    >>> expected = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    >>> a, b = 0, 10
    >>> Ya, Yb = [0, 1], []
    >>> np.allclose(solve_linear_recurrence(P, R, a, b, Ya, Yb), expected)
    True
    """
    Ya, Yb = np.array(Ya), np.array(Yb)
    m = Yb.size
    n = Ya.size + m
    nband = n + 1
    a = a - 1  # paper's convention
    nval = b - a - n
    l_and_u = (nval - (b - n - a - n + m), nval - (b - n - m - a))
    band = np.zeros((nband, nval))
    for i in range(m + 1):
        idx = np.arange(a + 1, b - n - m + 1 + i)
        band[i, -idx.size :] = P(i, idx)
    for i in range(m + 1, nband):
        idx = np.arange(a + n - m + 1 - (n - i), b - n + 1)
        band[i, : idx.size] = P(i, idx)
    rho = R(np.arange(a + 1, b - n + 1)).astype(float)
    rho[: n - m] -= [
        np.sum([P(n - s + 1, a + 1 + i) * Ya[s - 1 + i] for s in range(1, n - m + 1 - i)]) for i in range(n - m)
    ]
    if m > 0:
        rho[-m:] -= [np.sum([P(s, b - n - i) * Yb[-1 - s - i] for s in range(0, m - 1 + 1 - i)]) for i in range(m)][
            ::-1
        ]
    return solve_banded(l_and_u, band, rho)


class ChebyShevPolynomial:
    def __init__(self, n):
        self.n = n
        self.u = np.cos(np.pi * (np.arange(n) + 0.5) / n)

    def approximate(self, f, limits=(-1, 1)):
        self.f = f
        self.limits = limits
        scale = 0.5 * (limits[1] - limits[0])
        shift = 0.5 * (limits[1] + limits[0])
        self.x = self.u if tuple(limits) == (-1, 1) else scale * self.u + shift
        self.y = f(self.x)
        self.coef = scipy.fft.dct(self.y, type=2, norm=None) / self.n
        self.coef[0] /= 2.0
        self._poly = None
        return self

    def plot_comparison(self, ax=None, show=True):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        x = np.linspace(self.limits[0], self.limits[1], 1000)
        ax.plot(x, self.f(x), label="Original Function")
        ax.plot(x, self.poly(x), label="Chebyshev Approximation", linestyle="--")
        ax.legend()
        if show:
            plt.show()
        return ax

    @property
    def poly(self):
        if self._poly is None:
            self._poly = np.polynomial.Chebyshev(self.coef, domain=self.limits)
        return self._poly

    def __call__(self, x):
        return self.poly(x)
