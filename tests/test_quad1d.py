import pytest
import numpy as np
from pytest import approx
from scipy.integrate import fixed_quad

from quad4pt.quad1d import Trapezoidal, GaussLegendre, GaussChebyShev, ClenshawCurtis, DoubleExponential
from quad4pt.transforms import Logarithm


@pytest.mark.parametrize("n", [2, 5, 100, 500])
def test_trapezoidal(n):
    f = lambda x: np.sin(x)
    a, b = 0, np.pi
    integral = Trapezoidal(n)(f, [a, b])
    expected = np.trapezoid(f(np.linspace(a, b, n)), np.linspace(a, b, n))
    np.testing.assert_allclose(integral, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n", [2, 5, 100, 500])
def test_gauss_legendre(n):
    f = lambda x: np.sin(x)
    a, b = 0, np.pi
    expected = fixed_quad(f, a, b, n=n)[0]
    integral = GaussLegendre(n)(f, [a, b])
    np.testing.assert_allclose(integral, expected, rtol=1e-12, atol=1e-14)


def test_gauss_chebyshev():
    f = lambda x: np.ones_like(x)
    a, b = -1, 1
    expected = np.pi
    integral = GaussChebyShev(10)(f, [a, b])
    np.testing.assert_allclose(integral, expected, rtol=1e-12, atol=1e-14)


def test_clenshaw_curtis():
    f = lambda x: np.sin(x)
    a, b = 0, np.pi
    expected = 2
    integral = ClenshawCurtis(16)(f, [a, b])
    np.testing.assert_allclose(integral, expected, rtol=1e-12, atol=1e-14)

    a, b = 1e-3, np.pi
    expected = 1 + np.cos(1e-3)
    integral = ClenshawCurtis(64)(f, [a, b], transform=Logarithm())
    np.testing.assert_allclose(integral, expected, rtol=1e-12, atol=1e-14)


def test_gauss_legendre_with_logrithm_transform():
    f = lambda x: np.ones_like(x)
    a, b = 1e-5, 100.0
    expected = b - a
    integral = GaussLegendre(50)(f, [a, b], transform=Logarithm())
    np.testing.assert_allclose(integral, expected, rtol=1e-12, atol=1e-14)


def test_double_exponential():
    f = lambda x: np.log(1 / x) / x ** (1 / 4)
    a, b = 0, 1
    expected = 16 / 9
    integral = DoubleExponential(50)(f, [a, b])
    assert integral == approx(expected, rel=1e-10, abs=1e-12)
