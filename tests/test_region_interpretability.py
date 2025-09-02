# tests/test_region_interpretability.py
import numpy as np
import warnings
from sheshe.region_interpretability import _extract_region, _rdp, RegionInterpreter


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


def test_extract_region_missing_directions():
    reg = {
        "center": [0.0, 0.0],
        "radii": [1.0, 2.0],
    }
    cid, label, center, directions, radii = _extract_region(reg)
    assert cid == 0 and label == 0
    assert directions.shape == (4, 2)
    np.testing.assert_array_equal(radii, np.array([1.0, 2.0, 1.0, 2.0]))


def test_extract_region_inflection_fallback():
    center = [0.0, 0.0]
    infl = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -2.0]])
    reg = {"center": center, "inflection_points": infl}
    cid, label, center_out, directions, radii = _extract_region(reg)
    assert cid == 0 and label == 0
    np.testing.assert_array_almost_equal(center_out, np.array(center))
    expected_dirs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
    np.testing.assert_array_almost_equal(directions, expected_dirs)
    np.testing.assert_array_almost_equal(radii, np.array([1.0, 2.0, 1.0, 2.0]))


def test_summarize_with_inflection_only():
    center = [0.0, 0.0]
    infl = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    reg = {"center": center, "inflection_points": infl}
    card = RegionInterpreter(feature_names=["x", "y"]).summarize(reg)[0]
    assert card["headline"] != "Región degenerada (sin radios útiles)."
    assert card["box_rules"] == ["x ∈ [-0.8, 0.8]", "y ∈ [-0.8, 0.8]"]

def test_rdp_no_deprecation_warning():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simplified = _rdp(pts, epsilon=0.01)
    assert not any(issubclass(warn.category, DeprecationWarning) for warn in w)
    assert simplified.shape[1] == 2
    assert simplified.shape[0] >= 2
