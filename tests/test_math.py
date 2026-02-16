import numpy as np

from quad4pt.math import solve_linear_recurrence


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
