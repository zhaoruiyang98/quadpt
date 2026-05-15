import itertools
import numpy as np
from scipy.special import jn_zeros
from typing import Callable

from .math import spherical_jn_zeros
from .utils import import_x64_jax


class Kernel:
    kernel: Callable
    weight: Callable
    alternating = False

    def __call__(self, x):
        return self.kernel(x)

    def zeros_between(self, xmin, xmax):
        """Kernel function zeros in [xmin, xmax]."""
        raise NotImplementedError

    def integral_transform_expansion(self, y, limits, n):
        """Asymptotic expansion of the integral transform with the specific kernel function."""
        raise NotImplementedError


class CosineKernel(Kernel):
    def __init__(self, weight=lambda x: 1.0):
        self.kernel = np.cos
        self.weight = weight
        self.alternating = True

    def zeros_between(self, xmin, xmax):
        ileft = np.ceil(xmin / np.pi + 0.5).astype(int)
        iright = np.floor(xmax / np.pi + 0.5).astype(int)
        return (np.arange(ileft, iright + 1) - 0.5) * np.pi

    def integral_transform_expansion(self, y, limits, n):
        jax = import_x64_jax()
        a, b = limits
        va, vb = np.exp(1j * y * a), np.exp(1j * y * b)

        def expander(f, args=()):
            fn = lambda x: f(x, *args) * self.weight(x)
            results = 0.0
            for i in range(n):
                fna, fnb = fn(a), fn(b)
                coeff = -(1.0 / (-1.0j * y) ** (i + 1))
                ith_term = coeff * (fnb * vb - fna * va)
                results = results + ith_term.real
                fn = jax.grad(fn)
            return results

        return expander


class SineKernel(Kernel):
    def __init__(self, weight=lambda x: 1.0):
        self.kernel = np.sin
        self.weight = weight
        self.alternating = True

    def zeros_between(self, xmin, xmax):
        ileft = np.ceil(xmin / np.pi).astype(int)
        iright = np.floor(xmax / np.pi).astype(int)
        return np.arange(ileft, iright + 1) * np.pi

    def integral_transform_expansion(self, y, limits, n):
        jax = import_x64_jax()
        a, b = limits
        va, vb = np.exp(1j * y * a), np.exp(1j * y * b)

        def expander(f, args=()):
            fn = lambda x: f(x, *args) * self.weight(x)
            results = 0.0
            for i in range(n):
                fna, fnb = fn(a), fn(b)
                coeff = -(1.0 / (-1.0j * y) ** (i + 1))
                ith_term = coeff * (fnb * vb - fna * va)
                results = results + ith_term.imag
                fn = jax.grad(fn)
            return results

        return expander


class BesselKernel(Kernel):
    def __init__(self, nu, weight=lambda x: 1.0):
        from scipy.special import jv

        self.nu = nu
        self.kernel = lambda x: jv(self.nu, x)
        self.weight = weight
        self.alternating = True

    def zeros_between(self, xmin, xmax):
        if np.floor(self.nu) == self.nu - 1 / 2:
            # half-integer
            kernel = SphericalBesselKernel(np.floor(self.nu).astype(int))
            return kernel.zeros_between(xmin, xmax)
        if np.floor(self.nu) != self.nu:
            raise ValueError("Does not support non-integer orders.")
        # upper bound https://math.stackexchange.com/questions/1518358/zeros-of-bessel-functions
        nmax = np.ceil((xmax - 1) / np.pi + 1 / 4 - self.nu / 2).astype(int)
        zeros = jn_zeros(self.nu, nmax)
        ileft = np.searchsorted(zeros, xmin)
        iright = np.searchsorted(zeros, xmax, side="right")
        return zeros[ileft:iright]


class SphericalBesselKernel(Kernel):
    def __init__(self, nu, weight=lambda x: 1.0):
        from scipy.special import spherical_jn

        if np.floor(nu) != nu:
            raise ValueError("Does not support non-integer orders.")
        self.nu = nu
        self.kernel = lambda x: spherical_jn(self.nu, x)
        self.weight = weight
        self.alternating = True

    def zeros_between(self, xmin, xmax):
        return np.fromiter(
            itertools.takewhile(
                lambda z: z <= xmax,
                itertools.dropwhile(lambda z: z < xmin, spherical_jn_zeros(self.nu)),
            ),
            np.float64,
        )


class PowerToCorrelationKernel(SphericalBesselKernel):
    def __init__(self, nu, weight=lambda x: 1.0):
        from scipy.special import spherical_jn

        self.nu = nu
        coef = 1 / (2 * np.pi**2) * (1j) ** self.nu
        coef = coef.real if nu % 2 == 0 else coef.imag
        self.kernel = lambda x: spherical_jn(self.nu, x)
        self.weight = lambda x: x * x * coef * weight(x)
        self.alternating = True


class CorrelationToPowerKernel(SphericalBesselKernel):
    def __init__(self, nu, weight=lambda x: 1.0):
        from scipy.special import spherical_jn

        self.nu = nu
        coef = 4 * np.pi * (-1j) ** self.nu
        coef = coef.real if nu % 2 == 0 else coef.imag
        self.kernel = lambda x: spherical_jn(self.nu, x)
        self.weight = lambda x: x * x * coef * weight(x)
        self.alternating = True
