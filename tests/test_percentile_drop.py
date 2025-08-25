import numpy as np
import pytest

from sheshe.sheshe import find_percentile_drop


def test_percentile_drop_detects_lower_percentile_center_out():
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    vals = np.array([0.9, 0.85, 0.84, 0.5])
    percentiles = np.array([0.0, 0.8, 0.9, 1.0])

    t_drop, slope = find_percentile_drop(ts, vals, "center_out", percentiles, drop_fraction=0.5)

    assert t_drop == pytest.approx(1.0)
    assert slope == pytest.approx(-0.05, abs=1e-9)


def test_percentile_drop_detects_lower_percentile_outside_in():
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    vals = np.array([0.5, 0.84, 0.85, 0.9])
    percentiles = np.array([0.0, 0.8, 0.9, 1.0])

    t_drop, slope = find_percentile_drop(ts, vals, "outside_in", percentiles, drop_fraction=0.5)

    assert t_drop == pytest.approx(1.0)
    assert slope == pytest.approx(0.05, abs=1e-9)


def test_percentile_drop_uses_fraction_when_no_percentile_drop():
    ts = np.array([0.0, 1.0, 2.0])
    vals = np.array([1.0, 0.8, 0.4])
    percentiles = np.array([0.0, 2.0])  # single bin so no percentile drop

    t_drop, slope = find_percentile_drop(ts, vals, "center_out", percentiles, drop_fraction=0.5)

    assert t_drop == pytest.approx(1.75)
    assert slope == pytest.approx(-0.4)


def test_percentile_drop_invalid_direction():
    ts = np.array([0.0, 1.0])
    vals = np.array([1.0, 0.0])
    percentiles = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        find_percentile_drop(ts, vals, "invalid", percentiles)


@pytest.mark.parametrize("frac", [-0.1, 0.0, 1.0, 1.5])
def test_percentile_drop_invalid_drop_fraction(frac):
    ts = np.array([0.0, 1.0])
    vals = np.array([1.0, 0.0])
    percentiles = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        find_percentile_drop(ts, vals, "center_out", percentiles, drop_fraction=frac)
