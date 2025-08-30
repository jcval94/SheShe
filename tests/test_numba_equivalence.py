import numpy as np
import pytest

from sheshe.sheshe import (
    finite_diff_gradient,
    finite_diff_hessian,
    finite_diff_gradient_numba,
    finite_diff_hessian_numba,
    gradient_ascent,
    trust_region_newton,
)

pytest.importorskip("numba")
from numba import njit


def test_numba_finite_differences_match_python():
    def f_py(x):
        return np.sin(x[0]) + x[0] * x[1]

    @njit
    def f_nb(x):
        return np.sin(x[0]) + x[0] * x[1]

    x = np.array([0.2, -0.1])
    g_py = finite_diff_gradient(f_py, x, eps=1e-6)
    g_nb = finite_diff_gradient_numba(f_nb, x, eps=1e-6)
    h_py = finite_diff_hessian(f_py, x, eps=1e-6)
    h_nb = finite_diff_hessian_numba(f_nb, x, eps=1e-6)
    assert np.allclose(g_py, g_nb, atol=1e-6)
    assert np.allclose(h_py, h_nb, atol=1e-6)


def test_numba_and_python_algorithms_converge_same():
    @njit
    def f_nb(x):
        return -((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)

    def f_py(x):
        return -((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)

    x0 = np.array([5.0, 5.0])
    lo = np.array([-10.0, -10.0])
    hi = np.array([10.0, 10.0])

    res_nb = gradient_ascent(f_nb, x0, (lo, hi), lr=0.2, max_iter=500)
    res_py = gradient_ascent(f_py, x0, (lo, hi), lr=0.2, max_iter=500)
    assert np.allclose(res_nb, res_py, atol=1e-4)

    res_nb_newton = trust_region_newton(f_nb, x0, (lo, hi), trust_radius=10.0)
    res_py_newton = trust_region_newton(f_py, x0, (lo, hi), trust_radius=10.0)
    assert np.allclose(res_nb_newton, res_py_newton, atol=1e-4)
