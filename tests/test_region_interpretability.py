# tests/test_region_interpretability.py
import numpy as np
import warnings
from sheshe.region_interpretability import _extract_region, _rdp


def test_extract_region_label_na():
    reg = {
        "label": "NA",
        "center": [0.0, 0.0],
        "directions": np.eye(2),
        "radii": [1.0, 1.0],
    }
    cid, label, center, directions, radii = _extract_region(reg)
    assert cid == "NA"
    assert label == "NA"
    np.testing.assert_array_equal(center, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(directions, np.eye(2))
    np.testing.assert_array_equal(radii, np.array([1.0, 1.0]))


def test_extract_region_alt_keys():
    reg = {
        "cluster_id": 3,
        "label": 7,
        "center_": [1.0, 2.0],
        "directions_": np.eye(2),
        "radii_": [0.5, 0.25],
    }
    cid, label, center, directions, radii = _extract_region(reg)
    assert cid == 3
    assert label == 7
    np.testing.assert_array_equal(center, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(directions, np.eye(2))
    np.testing.assert_array_equal(radii, np.array([0.5, 0.25]))


def test_rdp_no_deprecation_warning():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simplified = _rdp(pts, epsilon=0.01)
    assert not any(issubclass(warn.category, DeprecationWarning) for warn in w)
    assert simplified.shape[1] == 2
    assert simplified.shape[0] >= 2
