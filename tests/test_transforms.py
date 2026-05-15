import numpy as np
from pytest import approx

from quadpt.transforms import Affine, Logarithm


def test_affine_transform():
    affine = Affine.regularize(2.0, 10.0)
    assert affine(2.0) == approx(-1.0)
    assert affine(10.0) == approx(1.0)
    assert affine.backward(-1.0) == approx(2.0)
    assert affine.backward(1.0) == approx(10.0)
    assert affine.jac(0.0) == approx(2.0 / (10.0 - 2.0))
    assert affine.jacinv(0.0) == approx(1 / affine.jac(0.0))

    affine2 = Affine.map_to([2.0, 10.0], [-1, 1])
    assert affine2.scale == approx(affine.scale)
    assert affine2.shift == approx(affine.shift)


def test_logarithm_transform():
    log = Logarithm()
    assert log(1.0) == approx(0.0)
    assert log(2.0) == approx(np.log(2.0))
    assert log(10.0) == approx(np.log(10.0))
    assert log.backward(0.0) == approx(1.0)
    assert log.backward(np.log(2.0)) == approx(2.0)
    assert log.jac(2.0) == approx(1 / 2.0)
    assert log.jacinv(0.0) == approx(1)


def test_composite_transform():
    affine = Affine(scale=2.0, shift=1.0)
    log = Logarithm()
    composite = affine @ log

    assert composite.forward(2.0) == approx(2.0 * np.log(2.0) + 1.0)
    assert composite.backward(10.2) == approx(np.exp((10.2 - 1.0) / 2.0))
    assert composite.jac(2.0) == approx(2.0 / 2.0)
    assert composite.jacinv(2.0) == approx(1 / 2 * np.exp((2.0 - 1.0) / 2))


def test_composite_affine_returns_affine():
    affine1 = Affine(scale=2.0, shift=1.0)
    affine2 = Affine(scale=3.0, shift=0.5)
    composite = affine1 @ affine2

    assert isinstance(composite, Affine)
    assert composite.scale == approx(6)
    assert composite.shift == approx(2)

    composite = type(affine1).chain(affine1, affine2)
    assert len(composite.transforms) == 1
    composite = composite.transforms[0]
    assert isinstance(composite, Affine)
    assert composite.scale == approx(6)
    assert composite.shift == approx(2)
