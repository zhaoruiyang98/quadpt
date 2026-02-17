import numpy as np
import pytest

from quad4pt.math import solve_linear_recurrence, itjv


def test_solve_linear_recurrence():
    P = lambda n, i: {0: -1, 1: 1, 2: 1}[n] * np.ones_like(i)
    R = lambda i: np.zeros_like(i)
    expected = [1, 2, 3, 5, 8, 13, 21, 34, 55]

    # initial value problem
    a = 0
    b = 10
    Ya = [0, 1]
    Yb = []
    result = solve_linear_recurrence(P, R, a, b, Ya, Yb)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0)

    # boundary value problem
    a = 0
    b = 11
    Ya = [0]
    Yb = [89]
    result = solve_linear_recurrence(P, R, a, b, Ya, Yb)
    np.testing.assert_allclose(result, [1] + expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "nu,expected",
    [
        (-1, [-0.2348023134420334, -1.245935764451348, -0.9800141496957769, -0.9752133138475798, -1.001719201116236]),
        (-1 / 2, [1.443411848585210, 0.8739279054587641, 0.9592570081448326, 1.020856156199700, 1.000090211562378]),
        (0, [0.9197304100897602, 1.067011303956737, 0.9226625569601661, 1.004703520567027, 1.001846774754709]),
        (1 / 2, [0.4951165753032217, 1.216872518130222, 0.9314039992875214, 0.9858000096769721, 1.002521519303296]),
    ],
)  # fmt: skip
def test_itjnu(nu, expected):
    x = [1, 10, 100, 1000, 100000]
    result = itjv(nu, x)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
