#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "fastgl.h"

namespace nb = nanobind;

using namespace nb::literals;

using ndarray1d = nb::ndarray<double, nb::shape<-1>>;

NB_MODULE(_fastgl, m) {
    m.def("leggauss", [](int n, ndarray1d const& x, ndarray1d const& w) {
        for (int i = 0; i < n; ++i) {
            fastgl::QuadPair p = fastgl::GLPair(n, i + 1);
            x(n - i - 1) = p.x();
            w(i) = p.weight;
        }
    });
}