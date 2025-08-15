import numpy as np
from sklearn.datasets import load_iris

from sheshe import SubspaceScout


def test_subspace_scout_basic():
    data = load_iris()
    X, y = data.data, data.target
    scout = SubspaceScout(max_order=2, top_m=4, sample_size=None, random_state=0)
    results = scout.fit(X, y)
    assert isinstance(results, list)
    assert results and all('features' in r for r in results)
