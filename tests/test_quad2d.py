import numpy as np
from pytest import approx

from quadpt.quad1d import Trapezoidal
from quadpt.quad2d import CartesianGridQuadrature


def test_cartesian_grid_integrator():
    f = lambda x, y: np.exp(-(x**2) - y**2)
    xlimits, ylimits = (-3, 3), (-3, 3)
    expected = 3.14145385643669
    trapz = Trapezoidal(100)
    integral = CartesianGridQuadrature([trapz, trapz])(f, xlimits, ylimits)
    assert integral == approx(expected, rel=1e-6, abs=1e-6)
