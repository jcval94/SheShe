import numpy as np

from sheshe.sheshe import (
    enforce_minimum_subspace,
    per_direction_step_lengths,
    cast_rays_with_censoring,
)


def test_enforce_minimum_subspace_fallback_verbose():
    X = np.random.randn(10, 3)
    dirs = enforce_minimum_subspace(None, X, k_fallback=2, verbose=1, log_fn=lambda m: None)
    assert dirs.shape == (4, 3)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0)
    axes = np.argmax(np.abs(dirs), axis=1)
    counts = np.bincount(axes, minlength=3)
    nz = counts[counts > 0]
    assert np.all(nz == 2)


def test_cast_rays_with_censoring_verbose():
    center = np.zeros(2)
    dirs = np.array([[1.0, 0.0]])
    std = np.ones(2)
    steps = per_direction_step_lengths(dirs, std)

    def f(x):
        return float(np.exp(-np.linalg.norm(x)))

    radii, pts, slopes, metrics = cast_rays_with_censoring(
        center=center,
        directions=dirs,
        eval_fn=f,
        bounds=None,
        max_steps=3,
        step_lengths=steps,
        min_drop=10.0,
        min_slope=0.0,
        verbose=1,
        log_fn=lambda m: None,
    )

    assert metrics["censored_mask"][0]
    assert np.isclose(radii[0], steps[0] * 3)
    assert np.allclose(pts[0], np.array([steps[0] * 3, 0.0]))
