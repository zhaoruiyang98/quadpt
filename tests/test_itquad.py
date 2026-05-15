import pytest
import numpy as np
from pytest import approx

from quadpt.itquad import AsymptoticExpansion, NodeByNode, FilonClenshawCurtis
from quadpt.utils import import_x64_jax

try:
    import jax  # noqa: F401

    jax_not_installed = False
except ImportError:
    jax_not_installed = True


@pytest.mark.skipif(jax_not_installed, reason="JAX not installed")
def test_asymptotic_expansion():
    jax = import_x64_jax()
    y = 100.0
    expected = (np.e**2 - np.cos(2 * y) + y * np.sin(2 * y)) / (np.e**2 * (1.0 + y**2))
    quad = AsymptoticExpansion.Cos(y=y, limits=(0.0, 2.0), n=5)
    integral = quad(lambda x: jax.numpy.exp(-x))
    assert integral == approx(expected, rel=1e-10, abs=1e-12)

    expected = -(-(np.e**2) * y + y * np.cos(2 * y) + np.sin(2 * y)) / (np.e**2 * (1.0 + y**2))
    quad = AsymptoticExpansion.Sin(y=y, limits=(0.0, 2.0), n=5)
    integral = quad(lambda x: jax.numpy.exp(-x))
    assert integral == approx(expected, rel=1e-10, abs=1e-12)


# fmt: off
cases = [
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 0, 1e0, 4.472135954999579e-1, id="I1=exp(-2x)*Jv(ax) v=0 a=1"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 0, 1e1, 9.805806756909202e-2, id="I1=exp(-2x)*Jv(ax) v=0 a=10"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 0, 1e2, 9.998000599800070e-3, id="I1=exp(-2x)*Jv(ax) v=0 a=100"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 0, 1e3, 9.999980000060000e-4, id="I1=exp(-2x)*Jv(ax) v=0 a=1000"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 0, 1e4, 9.999999800000006e-5, id="I1=exp(-2x)*Jv(ax) v=0 a=10000"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 2, 1e0, 2.492235949962145e-2, id="I1=exp(-2x)*Jv(ax) v=2 a=1"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 2, 1e1, 6.590271297461938e-2, id="I1=exp(-2x)*Jv(ax) v=2 a=10"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 2, 1e2, 9.605999000279910e-3, id="I1=exp(-2x)*Jv(ax) v=2 a=100"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 2, 1e3, 9.960059999900000e-4, id="I1=exp(-2x)*Jv(ax) v=2 a=1000"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 30), 2, 1e4, 9.996000599999990e-5, id="I1=exp(-2x)*Jv(ax) v=2 a=10000"),
]
cases += [
    pytest.param(lambda x: x * np.cos((1 - x**2) / 2), (0, 1), 0, 10, 4.286533241834777e-3, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=0 a=10"),
    pytest.param(lambda x: x * np.cos((1 - x**2) / 2), (0, 1), 0, 50, -1.950978701494231e-3, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=0 a=50"),
    pytest.param(lambda x: x * np.cos((1 - x**2) / 2), (0, 1), 0, 100, -7.715298117631174e-4, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=0 a=100"),
    pytest.param(lambda x: x**2 * np.cos((1 - x**2) / 2), (0, 1), 1, 10, 2.568245788763414e-2, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=1 a=10"),
    pytest.param(lambda x: x**2 * np.cos((1 - x**2) / 2), (0, 1), 1, 50, -1.194823022623969e-3, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=1 a=50"),
    pytest.param(lambda x: x**2 * np.cos((1 - x**2) / 2), (0, 1), 1, 100, -2.153136826074732e-4, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=1 a=100"),
    pytest.param(lambda x: x**3 * np.cos((1 - x**2) / 2), (0, 1), 2, 10, 6.074137505141018e-3, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=2 a=10"),
    pytest.param(lambda x: x**3 * np.cos((1 - x**2) / 2), (0, 1), 2, 50, 1.855347476820902e-3, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=2 a=50"),
    pytest.param(lambda x: x**3 * np.cos((1 - x**2) / 2), (0, 1), 2, 100, 7.629162199581855e-4, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=2 a=100"),
    pytest.param(lambda x: x**4 * np.cos((1 - x**2) / 2), (0, 1), 3, 10, -2.194265191220747e-2, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=3 a=10"),
    pytest.param(lambda x: x**4 * np.cos((1 - x**2) / 2), (0, 1), 3, 50, 1.417516846980848e-3, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=3 a=50"),
    pytest.param(lambda x: x**4 * np.cos((1 - x**2) / 2), (0, 1), 3, 100, 2.61091624195900373e-4, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=3 a=100"),
    pytest.param(lambda x: x**5 * np.cos((1 - x**2) / 2), (0, 1), 4, 10, -2.36199574622375093e-2, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=4 a=10"),
    pytest.param(lambda x: x**5 * np.cos((1 - x**2) / 2), (0, 1), 4, 50, -1.62848897053382527e-3, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=4 a=50"),
    pytest.param(lambda x: x**5 * np.cos((1 - x**2) / 2), (0, 1), 4, 100, -7.42027548660325501e-4, id="I2=x**(v+1)*cos(0.5*(1-x**2))*Jv(ax) v=4 a=100"),
]
cases += [
    pytest.param(lambda x: x*(1-x**2)**0.5, (0, 1), 0, 1, 3.011686789397568e-1, id="I3=x*(1-x**2)**0.5*J0(ax) v=0 a=1"),
    pytest.param(lambda x: x*(1-x**2)**0.5, (0, 1), 0, 2, 2.176988874899958e-1, id="I3=x*(1-x**2)**0.5*J0(ax) v=0 a=2"),
    pytest.param(lambda x: x*(1-x**2)**0.5, (0, 1), 0, 4, 2.902768731478937e-2, id="I3=x*(1-x**2)**0.5*J0(ax) v=0 a=4"),
    pytest.param(lambda x: x*(1-x**2)**0.5, (0, 1), 0, 8, 4.205778353695879e-3, id="I3=x*(1-x**2)**0.5*J0(ax) v=0 a=8"),
    pytest.param(lambda x: x*(1-x**2)**0.5, (0, 1), 0, 16, 3.670568449343039e-3, id="I3=x*(1-x**2)**0.5*J0(ax) v=0 a=16"),
    pytest.param(lambda x: x*(1-x**2)**0.5, (0, 1), 0, 32, -7.978430436696362e-4, id="I3=x*(1-x**2)**0.5*J0(ax) v=0 a=32"),
    pytest.param(lambda x: x*(1-x**2)**0.5, (0, 1), 0, 64, -9.215864833562626e-5, id="I3=x*(1-x**2)**0.5*J0(ax) v=0 a=64"),
    pytest.param(lambda x: x*(1-x**2)**0.5, (0, 1), 0, 128, 4.263482232870239e-5, id="I3=x*(1-x**2)**0.5*J0(ax) v=0 a=128"),
]
cases += [
    pytest.param(lambda x: x*(1-x**2)**1.5, (0, 1), 0, 1, 1.861051560341216e-1, id="I4=x*(1-x**2)**1.5*J0(ax) v=0 a=1"),
    pytest.param(lambda x: x*(1-x**2)**1.5, (0, 1), 0, 2, 1.488359617928599e-1, id="I4=x*(1-x**2)**1.5*J0(ax) v=0 a=2"),
    pytest.param(lambda x: x*(1-x**2)**1.5, (0, 1), 0, 4, 5.180319108212816e-2, id="I4=x*(1-x**2)**1.5*J0(ax) v=0 a=4"),
    pytest.param(lambda x: x*(1-x**2)**1.5, (0, 1), 0, 8, -5.205583395320395e-3, id="I4=x*(1-x**2)**1.5*J0(ax) v=0 a=8"),
    pytest.param(lambda x: x*(1-x**2)**1.5, (0, 1), 0, 16, 3.39910109057762096e-4, id="I4=x*(1-x**2)**1.5*J0(ax) v=0 a=16"),
    pytest.param(lambda x: x*(1-x**2)**1.5, (0, 1), 0, 32, -5.74969128510109519e-5, id="I4=x*(1-x**2)**1.5*J0(ax) v=0 a=32"),
    pytest.param(lambda x: x*(1-x**2)**1.5, (0, 1), 0, 64, -1.07313594666736327e-5, id="I4=x*(1-x**2)**1.5*J0(ax) v=0 a=64"),
    pytest.param(lambda x: x*(1-x**2)**1.5, (0, 1), 0, 128, -1.008032711116089602260e-6, id="I4=x*(1-x**2)**1.5*J0(ax) v=0 a=128"),
]
# fmt: on


@pytest.mark.parametrize("fn, limits, nu, a, expected", cases)
def test_filon_clenshaw_curtis_integral_transform_bessel(fn, limits, nu, a, expected):
    quad = FilonClenshawCurtis.Bessel(nu, y=a, limits=limits, n=4096)
    integral = quad(fn)
    assert integral == approx(expected, rel=1e-10, abs=1e-12)


# fmt: off
cases = [
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 0, 1e0, 4.191846148594207e-1, id="I1=exp(-2x)*jn(ax) n=0 a=1"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 0, 1e1, 1.385746856995352e-1, id="I1=exp(-2x)*jn(ax) n=0 a=10"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 0, 1e2, 1.549653668960441e-2, id="I1=exp(-2x)*jn(ax) n=0 a=100"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 0, 1e3, 1.568719884777387e-3, id="I1=exp(-2x)*jn(ax) n=0 a=1000"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 0, 1e4, 1.5706092140587316827e-4, id="I1=exp(-2x)*jn(ax) n=0 a=10000"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 0, 1e5, 1.5707776792812258037e-5, id="I1=exp(-2x)*jn(ax) n=0 a=100000"),

    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 2, 1e0, 5.2040160798763314467e-3, id="I1=exp(-2x)*jn(ax) n=2 a=1"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 2, 1e1, 4.5788046369396264307e-2, id="I1=exp(-2x)*jn(ax) n=2 a=10"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 2, 1e2, 7.4749687984329757631e-3, id="I1=exp(-2x)*jn(ax) n=2 a=100"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 2, 1e3, 7.8148368716433168383e-4, id="I1=exp(-2x)*jn(ax) n=2 a=1000"),
    pytest.param(lambda x: np.exp(-2 * x), (0, 1), 2, 1e4, 7.8498537161304524738e-5, id="I1=exp(-2x)*jn(ax) n=2 a=10000"),
    # pytest.param(lambda x: np.exp(-2 * x), (0, 1), 2, 1e5, 7.8535681185213626667e-6, id="I1=exp(-2x)*jn(ax) n=2 a=100000"),
]
# fmt: on


@pytest.mark.parametrize("fn, limits, n, a, expected", cases)
def test_node_by_node_integral_transform_spherical_bessel(fn, limits, n, a, expected):
    quad = NodeByNode.SphericalBessel(n, a, limits=limits, nmax=100, nmin=20)
    integral = quad(fn)
    assert integral == approx(expected, rel=1e-10, abs=1e-12)


# fmt: off
cases = [
    pytest.param(lambda x: x*np.exp(-2 * x), (0, 1), 0, 1e0, 1.398233212546408e-1, id="I1=x*exp(-2x)*jn(ax) n=0 a=1"),
    pytest.param(lambda x: x*np.exp(-2 * x), (0, 1), 0, 1e1, 1.084885608912617e-2, id="I1=x*exp(-2x)*jn(ax) n=0 a=10"),
    pytest.param(lambda x: x*np.exp(-2 * x), (0, 1), 0, 1e2, 8.843146880526663e-5, id="I1=x*exp(-2x)*jn(ax) n=0 a=100"),
    pytest.param(lambda x: x*np.exp(-2 * x), (0, 1), 0, 1e3, 9.236627618191649e-7, id="I1=x*exp(-2x)*jn(ax) n=0 a=1000"),

    pytest.param(lambda x: x*np.exp(-2 * x), (0, 1), 2, 1e0, 3.426847395730511e-3, id="I1=x*exp(-2x)*jn(ax) n=2 a=1"),
    pytest.param(lambda x: x*np.exp(-2 * x), (0, 1), 2, 1e1, 1.105753852228845e-2, id="I1=x*exp(-2x)*jn(ax) n=2 a=10"),
    pytest.param(lambda x: x*np.exp(-2 * x), (0, 1), 2, 1e2, 2.024761965933534e-4, id="I1=x*exp(-2x)*jn(ax) n=2 a=100"),
    pytest.param(lambda x: x*np.exp(-2 * x), (0, 1), 2, 1e3, 2.066589200941709e-6, id="I1=x*exp(-2x)*jn(ax) n=2 a=1000"),
]
# fmt: on


@pytest.mark.parametrize("fn, limits, nu, a, expected", cases)
def test_filon_clenshaw_curtis_integral_transform_spherical_bessel(fn, limits, nu, a, expected):
    # increasing n does not necessarily increase the precision
    quad = FilonClenshawCurtis.SphericalBessel(nu, y=a, limits=limits, n=4096)
    integral = quad(fn)
    assert integral == approx(expected, rel=1e-7, abs=1e-10)
