import numpy as np
from scipy.special import jv, jvp
from typing import Literal

from quad4pt.kernels import (
    Kernel,
    CosineKernel,
    SineKernel,
    BesselKernel,
    SphericalBesselKernel,
    PowerToCorrelationKernel,
    CorrelationToPowerKernel,
)
from quad4pt.math import ChebyShevPolynomial, itjv, solve_linear_recurrence
from quad4pt.quad1d import GaussLegendre


MethodT = Literal["asymptotic", "node-by-node", "Filon-Clenshaw-Curtis"]


class IntegralTransform:
    R"""Solve the following integral transform

    \int_a^b f(x) K(xy) dx

    lower bound a is usually required to be 0.
    """

    _subclasses = {}

    kernel: Kernel
    y: np.ndarray

    def __init__(self, kernel: Kernel, y, limits) -> None:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.name] = cls  # type: ignore

    def integrate(self, f, args=(), **kwargs):
        return self(f, args=args, **kwargs)

    def __call__(self, f, args=(), **kwargs):
        raise NotImplementedError

    @classmethod
    def Bessel(cls, integrand=None, method: MethodT = "Filon-Clenshaw-Curtis", **kwargs):
        tquad = cls._subclasses[method].Bessel(**kwargs)
        if integrand is None:
            return tquad
        return tquad(integrand)

    @classmethod
    def SphericalBessel(cls, integrand=None, method: MethodT = "node-by-node", **kwargs):
        tquad = cls._subclasses[method].SphericalBessel(**kwargs)
        if integrand is None:
            return tquad
        return tquad(integrand)

    @classmethod
    def P2xi(cls, integrand=None, method: MethodT = "node-by-node", **kwargs):
        tquad = cls._subclasses[method].P2xi(**kwargs)
        if integrand is None:
            return tquad
        return tquad(integrand)

    @classmethod
    def xi2P(cls, integrand=None, method: MethodT = "node-by-node", **kwargs):
        tquad = cls._subclasses[method].xi2P(**kwargs)
        if integrand is None:
            return tquad
        return tquad(integrand)

    @classmethod
    def Cos(cls, integrand=None, method: MethodT = "node-by-node", **kwargs):
        tquad = cls._subclasses[method].Cos(**kwargs)
        if integrand is None:
            return tquad
        return tquad(integrand)

    @classmethod
    def Sin(cls, integrand=None, method: MethodT = "node-by-node", **kwargs):
        tquad = cls._subclasses[method].Sin(**kwargs)
        if integrand is None:
            return tquad
        return tquad(integrand)


class AsymptoticExpansion(IntegralTransform):
    name = "asymptotic"

    def __init__(self, kernel: Kernel, y, limits, n) -> None:
        self.kernel = kernel
        self.squeeze = np.squeeze if np.ndim(y) == 0 else lambda x: x
        self.y = np.atleast_1d(y)
        self.limits = limits
        self.n = n
        self.expander = kernel.integral_transform_expansion(self.y, limits, n)

    def __call__(self, f, args=()):
        return self.squeeze(np.array(self.expander(f, args=args)))

    @classmethod
    def Cos(cls, y, limits, n):
        kernel = CosineKernel()
        return cls(kernel, y, limits, n)

    @classmethod
    def Sin(cls, y, limits, n):
        kernel = SineKernel()
        return cls(kernel, y, limits, n)


class NodeByNode(IntegralTransform):
    name = "node-by-node"

    def __init__(self, kernel: Kernel, y, limits, nmax=100, nmin=20, ntr=100) -> None:
        self.kernel = kernel
        self.squeeze = np.squeeze if np.ndim(y) == 0 else lambda x: x
        self.y = np.atleast_1d(y)
        self.limits = limits
        zmin, zmax = self.limits[0] * np.min(self.y), self.limits[1] * np.max(self.y)
        self.zeros = kernel.zeros_between(zmin, zmax)
        self.nmax = nmax
        self.nmin = nmin
        self.ntr = ntr
        self.quadmax = GaussLegendre(self.nmax)
        self.quadmin = GaussLegendre(self.nmin)

    @classmethod
    def Bessel(cls, nu, y, limits, **kwargs):
        kernel = BesselKernel(nu)
        return cls(kernel, y, limits, **kwargs)

    @classmethod
    def SphericalBessel(cls, nu, y, limits, **kwargs):
        kernel = SphericalBesselKernel(nu)
        return cls(kernel, y, limits, **kwargs)

    @classmethod
    def P2xi(cls, nu, s, limits, **kwargs):
        kernel = PowerToCorrelationKernel(nu)
        return cls(kernel, s, limits, **kwargs)

    @classmethod
    def xi2P(cls, nu, k, limits, **kwargs):
        kernel = CorrelationToPowerKernel(nu)
        return cls(kernel, k, limits, **kwargs)

    @classmethod
    def Cos(cls, y, limits, **kwargs):
        kernel = CosineKernel()
        return cls(kernel, y, limits, **kwargs)

    @classmethod
    def Sin(cls, y, limits, **kwargs):
        kernel = SineKernel()
        return cls(kernel, y, limits, **kwargs)

    def __call__(self, f, args=(), transform=None, show_progress=False):
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x: x  # type: ignore

        results = np.zeros_like(self.y, dtype=np.float64)
        xmin, xmax = self.limits
        y = tqdm(self.y) if show_progress else self.y
        for i, y in enumerate(y):
            zmin, zmax = xmin * y, xmax * y
            mask = (self.zeros >= zmin) & (self.zeros <= zmax)
            limits = self.zeros[mask] / y
            if limits.size == 0:
                lpad, rpad = [xmin], [xmax]
            else:
                lpad = [] if limits[0] == xmin else [xmin]
                rpad = [] if limits[-1] == xmax else [xmax]
            limits = np.concatenate([lpad, limits, rpad])
            quad = self.quadmax if limits.size - 1 <= self.ntr else self.quadmin
            quad = quad.bind(limits=limits, transform=transform)
            results[i] = quad(
                lambda x, args=args, y=y: f(x, *args) * self.kernel.weight(x) * self.kernel(x * y),
                alternating=self.kernel.alternating,
            )
        return self.squeeze(results)


class FilonClenshawCurtis(IntegralTransform):
    """Algorithm from Robert Piessens and Maria Branders' paper: https://doi.org/10.1007/BF01934465 .

    Warnings
    --------
    This implementation is not well tested when nu is half-integer.
    """

    name = "Filon-Clenshaw-Curtis"

    def __init__(self, kernel: Kernel, y, limits, n=64):
        if limits[0] != 0:
            raise ValueError("Filon-Clenshaw-Curtis Bessel Integral Transform does not support non-zero lower limit.")

        if not isinstance(
            kernel, (BesselKernel, SphericalBesselKernel, PowerToCorrelationKernel, CorrelationToPowerKernel)
        ):
            raise TypeError("Filon-Clenshaw-Curtis Bessel Integral Transform only supports Bessel-type kernels.")

        self.is_spherical = not isinstance(kernel, BesselKernel)
        self.kernel = kernel
        self.nu = self.kernel.nu
        self.squeeze = np.squeeze if np.ndim(y) == 0 else lambda x: x
        self.y = np.atleast_1d(y)
        self.limits = limits
        self.n = n
        # account for the difference between Jv and spherical_jv
        self.weight = (lambda x: np.sqrt(1 / x) * self.kernel.weight(x)) if self.is_spherical else self.kernel.weight
        self.postc = np.sqrt(np.pi / (2 * self.y)) if self.is_spherical else 1
        self.v = self.nu + (0.5 if self.is_spherical else 0)

        self.a = self.y * limits[1]  # limits -> [0, 1]
        self.poly = ChebyShevPolynomial(self.n)
        self.initialize()

    @classmethod
    def Bessel(cls, nu, y, limits, **kwargs):
        kernel = BesselKernel(nu)
        return cls(kernel, y, limits, **kwargs)

    @classmethod
    def SphericalBessel(cls, nu, y, limits, **kwargs):
        kernel = SphericalBesselKernel(nu)
        return cls(kernel, y, limits, **kwargs)

    @classmethod
    def P2xi(cls, nu, s, limits, **kwargs):
        kernel = PowerToCorrelationKernel(nu)
        return cls(kernel, s, limits, **kwargs)

    @classmethod
    def xi2P(cls, nu, k, limits, **kwargs):
        kernel = CorrelationToPowerKernel(nu)
        return cls(kernel, k, limits, **kwargs)

    def __call__(self, f, args=()):
        # coefficient b compensates the jacobian (limits -> [0, 1])
        self.poly.approximate(lambda x, b=self.limits[1]: b * f(x, *args) * self.weight(x), limits=self.limits)
        return self.squeeze((self.poly.coef @ self.M) * self.postc)

    def initialize(self):
        v = self.v
        a = self.a
        a2 = a * a
        a3 = a2 * a
        ainv = 1 / a
        ainv2 = ainv * ainv
        ainv3 = ainv2 * ainv

        G_0_vm1 = itjv(v - 1, a)
        G_0_v = itjv(v, a)
        J_vm1 = jv(v - 1, a)
        J_v = jv(v, a)

        G_1_v = v * G_0_vm1 - a * J_vm1
        G_2_v = (v**2 - 1) * G_0_v + (v + 1) * a * J_v - a2 * J_vm1
        G_3_v = (v**2 - 2**2) * G_1_v + (1 + v + 1) * a2 * J_v - a3 * J_vm1

        M_0 = ainv * G_0_v
        M_1 = M_m1 = ainv * (2 * ainv * G_1_v - G_0_v)
        M_2 = M_m2 = ainv * (8 * ainv2 * G_2_v - 8 * ainv * G_1_v + G_0_v)
        M_3 = M_m3 = ainv * (32 * ainv3 * G_3_v - 48 * ainv2 * G_2_v + 18 * ainv * G_1_v - G_0_v)

        M_4 = (
            -3 * (16 + a2 - 16 * v**2) * ainv2 * M_0
            - 64 * (1 + v**2) * ainv2 * M_1
            + (4 + 16 * (-9 + v**2) * ainv2) * M_2
        )
        M = [M_m3, M_m2, M_m1, M_0, M_1, M_2, M_3, M_4]

        if self.n <= 5:
            # already solved
            self.M = np.array(M[3 : 3 + self.n])
            return

        self.M = np.zeros((self.n, a.size))
        stable_case = self.n <= a / 2
        self.M[:, stable_case] = self.forward_recursion(v, a[stable_case], [_[stable_case] for _ in M])
        self.M[:, ~stable_case] = self.solve_bvp(v, a[~stable_case], [_[~stable_case] for _ in M])

    def forward_recursion(self, v, a, M):
        a2 = a * a
        ainv2 = 1 / a2
        M = M.copy()
        for k in range(1, self.n - 4):
            M_kp2, M_kp1, M_k, M_km1, M_km2, M_km4 = M[-2], M[-3], M[-4], M[-5], M[-6], M[-8]
            val1 = -6 * M_km2 + 6 * M_kp2 + 2 * M_kp1 - 2 * M_km1
            w1 = np.sign(val1)
            val2 = -2 * M_k + M_km2 + M_kp2
            w2 = np.sign(val2)
            Mnext = (
                -16
                * ainv2
                * (
                    (-(v**2) - a2 / 4) * M_kp2
                    + (4 * v**2 + 4) * M_kp1
                    - (-6 + 6 * v**2 - 3 / 8 * a2) * M_k
                    + (4 * v**2 + 4) * M_km1
                    + (-(v**2) - a2 / 4) * M_km2
                    + a2 / 16 * M_km4
                    + 9 * (M_km2 + M_kp2)
                    + w1 * np.exp(np.log(k) + np.log(np.abs(val1)))
                    + w2 * np.exp(2 * np.log(k) + np.log(np.abs(val2)))  # slightly reduces the rounding error
                )
            )
            M.append(Mnext)
        return np.array(M[3:])

    def solve_bvp(self, v, avec, Mvec):
        results = []
        for i, a in enumerate(avec):
            M = [_[i] for _ in Mvec]
            ell = max(self.n, np.ceil(a + 10).astype(int))  # +10 padding...
            M_l, M_lp1 = self.endpoint_condition(ell, v, a)
            a2 = a * a
            coef = {
                0: lambda k: a2 / 16,
                1: lambda k: np.zeros_like(k),
                2: lambda k: (k + 3) ** 2 - v**2 - a2 / 4,
                3: lambda k: (4 * v**2 + 2 * k + 4),
                4: lambda k: -(2 * k**2 - 6 + 6 * v**2 - 3 / 8 * a2),
                5: lambda k: 4 * v**2 - 2 * k + 4,
                6: lambda k: ((k - 3) ** 2 - v**2 - a2 / 4),
                7: lambda k: np.zeros_like(k),
                8: lambda k: a2 / 16,
            }
            # i = k - 4, standard form: P_0 * M[i + 8] + P_1 * M[i + 7] + ... + P_8 * M[i] = 0
            P = lambda n, i: coef[n](i + 4)
            R = lambda k: np.zeros_like(k)
            Msolved = solve_linear_recurrence(P, R, -2, ell + 1, M[1:-1], [M_l, M_lp1])
            results.append(np.hstack([M[3:-1], Msolved[: self.n - 4]]))
        return np.asarray(results).T

    def endpoint_condition(self, ell, v, a):
        a2 = a * a
        a3 = a2 * a
        if self.is_spherical:
            # derivative diverges at the third order, use direct integration instead
            # XXX: unfortunately this step may dominate the initialization time of this class
            calculate = lambda k: GaussLegendre(50).bind(limits=self.master_integrand_zeros(k, v, a))(
                lambda t: 0.5 * np.sin(t) * jv(v, (np.cos(t) + 1) * a / 2) * np.cos(k * t)
            )
            M_l = calculate(ell)
            M_lp1 = calculate(ell + 1)
        else:
            phi_D1_pi = -jv(v, 0)
            phi_D3_pi = jv(v, 0) - 3 / 2 * a * jvp(v, 0, n=1)
            phi_D1_0 = jv(v, a)
            phi_D3_0 = -jv(v, a) - 3 / 2 * a * jvp(v, a, n=1)
            phi_D5_pi = -jv(v, 0) + 15 / 2 * a * jvp(v, 0, n=1) - 15 / 4 * a2 * jvp(v, 0, n=2)
            phi_D5_0 = jv(v, a) + 15 / 2 * a * jvp(v, a, n=1) + 15 / 4 * a2 * jvp(v, a, n=2)
            phi_D7_pi = (
                jv(v, 0) - 63 / 2 * a * jvp(v, 0, n=1) + 105 / 2 * a2 * jvp(v, 0, n=2) - 105 / 8 * a3 * jvp(v, 0, n=3)
            )
            phi_D7_0 = (
                -jv(v, a) - 63 / 2 * a * jvp(v, a, n=1) - 105 / 2 * a2 * jvp(v, a, n=2) - 105 / 8 * a3 * jvp(v, a, n=3)
            )
            phi_D_pi = np.array([phi_D1_pi, phi_D3_pi, phi_D5_pi, phi_D7_pi])
            phi_D_0 = np.array([phi_D1_0, phi_D3_0, phi_D5_0, phi_D7_0])
            j = np.array([0, 1, 2, 3])
            # asymptotic expression of M_k(a, v)
            calculate = lambda k: 0.5 * np.sum((-1) ** j * ((-1) ** k * phi_D_pi - phi_D_0) * (1.0 * k) ** (-2 * j - 2))
            M_l = calculate(ell)
            M_lp1 = calculate(ell + 1)
        return M_l, M_lp1

    @classmethod
    def master_integrand_zeros(cls, k, v, a):
        zeros1 = CosineKernel().zeros_between(0, np.pi * k) / k
        zeros2 = np.arccos(BesselKernel(v).zeros_between(-a, a) * 2 / a - 1)
        return np.unique(np.hstack([0.0, zeros1, zeros2, np.pi]))
