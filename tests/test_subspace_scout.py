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


def test_subspace_scout_returns_default_for_two_dims():
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, size=50)
    scout = SubspaceScout(random_state=0)
    res = scout.fit(X, y)
    assert res == [{
        'features': (0, 1),
        'order': 2,
        'score': 1.0,
        'metric': 'mi_synergy'
    }]


def test_discretize_falls_back_without_quantile_method(monkeypatch):
    import sheshe.subspace_scout as sc
    from sklearn.preprocessing import KBinsDiscretizer

    class OldKBinsDiscretizer(KBinsDiscretizer):
        """Simulate an older scikit-learn without ``quantile_method``."""

        def __init__(
            self,
            n_bins=5,
            *,
            encode="onehot",
            strategy="quantile",
            dtype=None,
            subsample=200000,
            random_state=None,
        ):
            # The parent class accepts ``quantile_method``.  By omitting it from
            # the signature we emulate a version of scikit-learn where passing
            # that argument would raise ``TypeError``.
            super().__init__(
                n_bins=n_bins,
                encode=encode,
                strategy=strategy,
                dtype=dtype,
                subsample=subsample,
                random_state=random_state,
            )

    monkeypatch.setattr(sc, "KBinsDiscretizer", OldKBinsDiscretizer)
    scout = sc.SubspaceScout(random_state=0)
    X = np.random.rand(30, 3)
    y = np.random.randint(0, 2, size=30)
    # Should not raise even though the patched discretizer rejects
    # ``quantile_method`` when present.
    scout.fit(X, y)


def test_subspace_scout_returns_default_for_one_dim():
    X = np.random.rand(50, 1)
    y = np.random.randint(0, 2, size=50)
    scout = SubspaceScout(random_state=0)
    res = scout.fit(X, y)
    assert res == [{
        'features': (0,),
        'order': 1,
        'score': 1.0,
        'metric': 'mi_synergy'
    }]
