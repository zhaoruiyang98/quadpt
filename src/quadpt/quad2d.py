import itertools
import numpy as np
from collections.abc import Sequence
from .quad1d import BaseQuadrature, TransformedQuadrature, GaussLegendre, DoubleExponential
from .transforms import Affine


class CartesianGridQuadrature:
    def __init__(self, quads: Sequence[BaseQuadrature]):
        self.quads = list(quads)
        self.ndim = len(self.quads)

    def integrate(self, f, *limits, args=(), transform: list | None = None):
        return self(f, *limits, args=args, transform=transform)

    def __call__(self, f, *limits, args=(), transform: list | None = None):
        if transform is None:
            transform = [None] * self.ndim
        if len(transform) != self.ndim:
            raise ValueError("Length of transform should match the number of quadratures")

        limits = [list(itertools.pairwise(lim)) for lim in limits]
        return sum(self.one_block(f, limits, args, transform) for limits in itertools.product(*limits))

    def one_block(self, f, limits, args, transform):
        ws = []
        xs = []
        jacinvs = []

        for (i, quad), transf, lim in zip(enumerate(self.quads), transform, limits):
            if transf is None:
                transf = Affine.regularize(*lim)
            else:
                transf = Affine.regularize(*(transf(_) for _ in lim)) @ transf

            u = quad.u
            w = quad.w
            x = transf.backward(u)
            jacinv = transf.jacinv(u)

            shape = [1] * self.ndim
            shape[i] = -1
            xs.append(x.reshape(shape))
            ws.append(w.reshape(shape))
            jacinvs.append(np.asarray(jacinv).reshape(shape))

        weight = np.ones(1)
        for w, jacinv in zip(ws, jacinvs):
            weight = weight * w * jacinv

        values = f(*xs, *args)
        return np.sum(weight * values)

    def bind(self, *, transform=None):
        return TransformedQuadrature(self, transform)


class GaussLegendre2D:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.quad = CartesianGridQuadrature([GaussLegendre(nx), GaussLegendre(ny)])

    def integrate(self, f, xlimits, ylimits, *, args=(), transform: list | None = None):
        return self(f, xlimits=xlimits, ylimits=ylimits, args=args, transform=transform)

    def __call__(self, f, xlimits, ylimits, *, args=(), transform: list | None = None):
        return self.quad(f, xlimits, ylimits, args=args, transform=transform)

    def bind(self, *, transform=None):
        return TransformedQuadrature(self, transform)


class DoubleExponential2D:
    def __init__(self, nx, ny, txmax=3.172, tymax=3.172):
        self.nx = nx
        self.ny = ny
        self.quad = CartesianGridQuadrature([DoubleExponential(nx, tmax=txmax), DoubleExponential(ny, tmax=tymax)])

    def integrate(self, f, xlimits, ylimits, *, args=(), transform: list | None = None):
        return self(f, xlimits=xlimits, ylimits=ylimits, args=args, transform=transform)

    def __call__(self, f, xlimits, ylimits, *, args=(), transform: list | None = None):
        return self.quad(f, xlimits, ylimits, args=args, transform=transform)

    def optimize(self, xlimits, ylimits, epsxpad=0.0, epsypad=0.0):
        quad = CartesianGridQuadrature(
            [q.optimize(lim, epspad) for q, lim, epspad in zip(self.quad.quads, [xlimits, ylimits], [epsxpad, epsypad])]  # type: ignore
        )
        obj = self.__new__(type(self))
        obj.nx = self.nx
        obj.ny = self.ny
        obj.quad = quad
        return obj

    def bind(self, *, transform=None):
        return TransformedQuadrature(self, transform)


class TanhSinh2D(DoubleExponential2D):
    pass
