import itertools
import mpmath as mp
import numpy as np
import warnings
from typing import Any, Literal

from .math import ChebyShevPolynomial
from .transforms import Affine, Transform


class BaseQuadrature:
    n: int
    u: np.ndarray
    w: np.ndarray

    def integrate(self, f, limits, *, args=(), transform: Transform | None = None):
        return self(f, limits, args=args, transform=transform)

    def __call__(self, f, limits, *, args=(), transform: Transform | None = None):
        return sum(self.one_block(f, limits, args, transform) for limits in itertools.pairwise(limits))

    def one_block(self, f, limits, args, transform):
        if transform is None:
            transform = Affine.regularize(*limits)
        else:
            transform = Affine.regularize(*(transform(lim) for lim in limits)) @ transform
        u = self.u
        w = self.w
        x = transform.backward(u)
        jacinv = transform.jacinv(u)
        y = f(x, *args)
        return np.sum(jacinv * w * y)

    def bind(self, *, limits=None, transform=None) -> Any:
        if limits is not None:
            return MultiPanelQuadrature(self, limits, transforms=transform)
        return TransformedQuadrature(self, transform)


class TransformedQuadrature:
    def __init__(self, quad, transform):
        self.base = quad
        self.transform = transform

    def integrate(self, f, limits, *, args=()):
        return self(f, limits, args=args)

    def __call__(self, f, *limits, args=()):
        return self.base(f, *limits, args=args, transform=self.transform)


class MultiPanelQuadrature:
    def __init__(self, quads, limits, transforms=None):
        self.npanels = len(limits) - 1
        if not isinstance(quads, (list, tuple)):
            quads = [quads] * self.npanels
        if transforms is not None and not isinstance(transforms, (list, tuple)):
            transforms = [transforms] * self.npanels
        if transforms is None:
            transforms = [Affine.regularize(*lim) for lim in itertools.pairwise(limits)]
        else:
            transforms = [
                Affine.regularize(*(transf(_) for _ in lim)) @ transf
                for lim, transf in zip(itertools.pairwise(limits), transforms)
            ]
        if len(quads) != self.npanels or len(transforms) != self.npanels:
            raise ValueError("Number of quadratures must match number of panels, i.e., len(limits) - 1.")
        self.quads = quads
        self.limits = limits
        self.transforms = transforms
        # XXX: is it possible to speed up the following two lines?
        self.x = np.hstack([transf.backward(quad.u) for quad, transf in zip(self.quads, self.transforms)])
        self.w = np.hstack([transf.jacinv(quad.u) * quad.w for quad, transf in zip(self.quads, self.transforms)])

    def integrate(self, f, *, args=(), alternating=False):
        return self(f, args=args, alternating=alternating)

    def __call__(self, f, *, args=(), alternating=False):
        w = self.w
        v = f(self.x, *args)
        if alternating:
            return np.inner(w[::2], v[::2]) + np.inner(w[1::2], v[1::2])
        return np.inner(self.w, v)


class Trapezoidal(BaseQuadrature):
    def __init__(self, n):
        if n < 2:
            raise ValueError("Trapezoidal rule requires at least 2 points.")
        self.n = n
        self.u = np.linspace(-1, 1, n)
        self.w = np.ones(n)
        self.w[0] /= 2
        self.w[-1] /= 2
        self.w[...] *= 2 / (self.n - 1)


class Simpson(BaseQuadrature):
    def __init__(self, n):
        if n < 6:
            raise ValueError("Simpson's rule requires at least 6 points.")
        self.n = n
        self.u = np.linspace(-1, 1, n)
        self.w = np.ones(n)
        self.w[:3] = [3 / 8, 7 / 6, 23 / 24]
        self.w[-3:] = [23 / 24, 7 / 6, 3 / 8]
        self.w[...] *= 2 / (self.n - 1)


class GaussLegendre(BaseQuadrature):
    def __init__(self, n, backend: Literal["fastgl", "numpy"] = "fastgl"):
        self.n = n
        if backend == "fastgl":
            from .fastgl import leggauss
        else:
            leggauss = np.polynomial.legendre.leggauss
        self.u, self.w = leggauss(n)


class GaussChebyShev(BaseQuadrature):
    def __init__(self, n):
        self.n = n
        self.u, self.w = np.polynomial.chebyshev.chebgauss(n)


class ClenshawCurtis(BaseQuadrature):
    def __init__(self, n):
        self.poly = ChebyShevPolynomial(n)
        self.n = n
        self.u = self.poly.u
        k = np.arange(n // 2)
        self.w = -2 / ((2 * k + 1) * (2 * k - 1))

    def one_block(self, f, limits, args, transform):
        if transform is None:
            transform = Affine.regularize(*limits)
        else:
            transform = Affine.regularize(*(transform(lim) for lim in limits)) @ transform
        self.poly.approximate(lambda u: f(transform.backward(u), *args) * transform.jacinv(u))
        return np.inner(self.w, self.poly.coef[::2])

    def bind(self, *, limits=None, transform=None) -> Any:
        if limits is not None:
            raise NotImplementedError("MultiPanelQuadrature is not implemented for Clenshaw-Curtis.")
        return TransformedQuadrature(self, transform)


class DoubleExponential(BaseQuadrature):
    tmax_float = 4.026
    tmax_double = 6.112
    tmax_longdouble = 8.885

    tmax_double_one = 3.172

    def __init__(self, n, tmax: float | tuple[float, float] | list[float] = 3.172):
        if n % 2 == 0:
            n -= 1
        self.n = n
        if isinstance(tmax, (int, float)):
            tmax = (tmax, tmax)
        self.tmax = tmax = (abs(tmax[0]), abs(tmax[1]))
        if any(t > self.tmax_double for t in self.tmax):
            warnings.warn(f"tmax > {self.tmax_double} underflows in double precision")
        self.h = sum(self.tmax) / (n - 1)  # maximal spacing, from http://arxiv.org/abs/2007.15057

        half_pi = 0.5 * np.pi
        self.t = np.arange(0.0, self.h * (self.n - 1) + self.h / 2, self.h) - self.tmax[0]
        self.uc = np.exp(-np.sign(self.t) * half_pi * np.sinh(self.t)) / np.cosh(half_pi * np.sinh(self.t))
        self.u = np.sign(self.t) - np.sign(self.t) * self.uc  # lose precision
        cosh_v = np.cosh(half_pi * np.sinh(self.t))
        self._w = half_pi * np.cosh(self.t) / cosh_v / cosh_v
        self.w = self._w * self.h  # may lose precision

    def one_block(self, f, limits, args, transform):
        if transform is None:
            transform = Affine.regularize(*limits)
        else:
            transform = Affine.regularize(*(transform(lim) for lim in limits)) @ transform
        u = self.u
        uc = self.uc
        w = self._w
        result = np.zeros_like(self.u, dtype=np.float64)

        def run(mask, u, transform):
            x = transform.backward(u[mask])
            jacinv = transform.jacinv(u[mask])
            y = f(x, *args)
            result[mask] = jacinv * w[mask] * y

        run(u < -0.5, uc, Affine.map_to([-1, 0], [0, 1]) @ transform)
        run((u >= -0.5) & (u <= 0.5), u, transform)
        run(u > 0.5, uc, Affine.map_to([0, 1], [1, 0]) @ transform)
        result[u > 0.5] *= -1  # incorrect jacinv in the last line

        return self.h * np.sum(result)

    def optimize(self, limits, epspad=0.0):
        left, right = np.nextafter(limits[0], limits[1]), np.nextafter(limits[1], limits[0])
        leps, reps = left - limits[0], limits[1] - right
        lepspad, repspad = [epspad, epspad] if isinstance(epspad, (int, float)) else epspad
        leps, reps = lepspad + leps, repspad + reps
        tmax = [np.arcsinh(1 / np.pi * (-1 + np.log(2) - np.log(eps))) for eps in (leps, reps)]
        tmax = [min(t, self.tmax_double) for t in tmax]
        return type(self)(self.n, tmax=tmax)

    def bind(self, *, limits=None, transform=None) -> Any:
        if limits is not None:
            raise NotImplementedError("MultiPanelQuadrature is not implemented for DoubleExponential.")
        return TransformedQuadrature(self, transform)


class TanhSinh(DoubleExponential):
    pass


def mp_double_exponential_quad(f, a, b, n=40, tmax="6.112", ctx=mp.mp):
    if n % 2 == 0:
        n = n - 1
    n = n // 2
    total = ctx.mpf(0)
    h = ctx.mpf(tmax) / n
    a, b = ctx.mpf(a), ctx.mpf(b)
    for k in range(-n, n + 1):
        t = k * h
        sinh_t = ctx.sinh(t)
        cosh_t = ctx.cosh(t)
        u = ctx.pi / 2 * sinh_t
        tanh_u = ctx.tanh(u)
        sech2_u = 1 / ctx.cosh(u) ** 2
        x = (b - a) / 2 * tanh_u + (a + b) / 2
        dx_dt = (b - a) / 2 * (ctx.pi / 2) * cosh_t * sech2_u
        total += f(x) * dx_dt * h
    return total
