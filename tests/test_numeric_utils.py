import math
import numpy as np

from sheshe.sheshe import (
    generate_directions,
    rays_count_auto,
    find_inflection,
    gradient_ascent,
)


def test_generate_directions_formats_and_norms():
    # dim=1
    dirs1 = generate_directions(1, base_2d=8, random_state=0)
    assert dirs1.shape == (1, 1)
    assert np.allclose(np.linalg.norm(dirs1, axis=1), 1.0)

    # dim=2
    dirs2 = generate_directions(2, base_2d=8, random_state=0)
    assert dirs2.shape == (rays_count_auto(2, 8), 2)
    assert np.allclose(np.linalg.norm(dirs2, axis=1), 1.0)

    # dim=3
    dirs3 = generate_directions(3, base_2d=8, random_state=0)
    assert dirs3.shape == (rays_count_auto(3, 8), 3)
    assert np.allclose(np.linalg.norm(dirs3, axis=1), 1.0)

    # dim=5 (>3)
    dim = 5
    dirs5 = generate_directions(dim, base_2d=8, random_state=0)
    expected = rays_count_auto(dim, 8) + math.comb(dim, 3) * rays_count_auto(3, 8)
    assert dirs5.shape == (expected, dim)
    assert np.allclose(np.linalg.norm(dirs5, axis=1), 1.0)


def test_find_inflection_center_out():
    ts = np.linspace(0, 3.0, 501)
    vals = 1.0 / (1.0 + ts**2)
    t_inf, slope = find_inflection(ts, vals, "center_out")
    t_expected = 1.0 / math.sqrt(3.0)
    slope_expected = -2 * t_expected / (1 + t_expected**2) ** 2
    assert abs(t_inf - t_expected) < 1e-2
    assert abs(slope - slope_expected) < 1e-2


def test_find_inflection_outside_in():
    ts = np.linspace(0, 3.0, 301)
    vals = -(ts - 2.0) ** 3
    t_inf, slope = find_inflection(ts, vals, "outside_in")
    assert abs(t_inf - 1.0) < 1e-2
    assert abs(slope) < 1e-3


def test_find_inflection_with_smoothing():
    ts = np.linspace(0, 3.0, 301)
    base = 1.0 / (1.0 + ts**2)
    noise = 0.05 * np.sin(40 * ts)
    vals = base + noise
    t_expected = 1.0 / math.sqrt(3.0)
    t_raw, _ = find_inflection(ts, vals, "center_out")
    t_smooth, _ = find_inflection(ts, vals, "center_out", smooth_window=11)
    assert abs(t_smooth - t_expected) < abs(t_raw - t_expected)


def test_gradient_ascent_quadratic_convergence():
    def f(x):
        return -((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)

    def grad_f(x):
        return np.array([-2.0 * (x[0] - 1.0), -2.0 * (x[1] + 2.0)])

    x0 = np.array([5.0, 5.0])
    lo = np.array([-10.0, -10.0])
    hi = np.array([10.0, 10.0])
    res = gradient_ascent(f, x0, (lo, hi), lr=0.2, max_iter=500, gradient=grad_f)
    assert np.allclose(res, np.array([1.0, -2.0]), atol=5e-2)
