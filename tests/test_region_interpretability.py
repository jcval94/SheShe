# tests/test_region_interpretability.py
import numpy as np
from sheshe.region_interpretability import _extract_region


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
