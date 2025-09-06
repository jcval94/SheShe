import numpy as np

from sheshe import CheChe


def test_feature_selection_reduces_dims_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 10))
    y = X[:, 0] + 0.1 * rng.normal(size=100)
    ch = CheChe(random_state=0).fit(X, y)
    assert ch.selected_features_ is not None
    assert len(ch.selected_features_) < X.shape[1]
    preds = ch.predict(X)
    assert preds.shape[0] == X.shape[0]
