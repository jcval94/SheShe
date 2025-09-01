import numpy as np

from sheshe import CheChe, RegionInterpreter


def test_cheche_frontier_basic():
    X = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
    ])
    # only one pair has non-zero area; limit to that pair
    ch = CheChe().fit(X, max_pairs=1)
    assert set(ch.frontiers_.keys()) == {(0, 1)}
    frontier = ch.get_frontier((0, 1))
    assert frontier.shape[1] == 2
    expected = {(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)}
    assert expected.issubset(set(map(lambda p: tuple(np.round(p, 6)), frontier)))

    # regions should expose data compatible with RegionInterpreter
    assert len(ch.regions_) == 1
    reg = ch.regions_[0]
    for key in ["center", "directions", "radii", "frontier", "dims"]:
        assert key in reg
    assert reg["center"].shape == (2,)
    assert reg["directions"].shape[1] == 2
    assert reg["directions"].shape[0] == reg["radii"].shape[0]

    # RegionInterpreter should handle the region
    cards = RegionInterpreter(feature_names=["x0", "x1"]).summarize([reg])
    assert cards[0]["cluster_id"] == reg["cluster_id"]


def test_cheche_multiclass_frontiers():
    # two classes forming disjoint squares in the (0,1) plane
    X0 = np.array(
        [
            [0.2, 0.2],
            [0.2, 0.8],
            [0.8, 0.8],
            [0.8, 0.2],
        ]
    )
    X1 = X0 + 2.0  # shift to a different region
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))

    ch = CheChe().fit(X, y, feature_names=["f1", "f2"])

    assert ch.mode_ == "multiclass"
    assert set(ch.classes_) == {0, 1}
    # retrieve frontiers for each class
    f0 = ch.get_frontier((0, 1), class_index=0)
    f1 = ch.get_frontier((0, 1), class_index=1)
    assert f0.shape[1] == 2 and f1.shape[1] == 2


def test_cheche_regression_frontiers():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.random(100), rng.random(100), np.zeros(100)])
    y = X[:, 0] + X[:, 1]
    ch = CheChe().fit(X, y, max_pairs=1)
    assert ch.mode_ == "regression"
    assert len(ch.per_class_) == 10
    f0 = ch.get_frontier((0, 1), class_index=0)
    assert f0.shape[1] == 2


def test_mapping_level_subsamples_points():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.8, 0.0],
            [1.0, 1.0, 0.0],
            [0.8, 0.9, 0.0],
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ]
    )
    ch_mapped = CheChe().fit(X, max_pairs=1, mapping_level=2)
    ch_sub = CheChe().fit(X[::2], max_pairs=1)
    assert set(ch_mapped.frontiers_.keys()) == {(0, 1)}
    np.testing.assert_allclose(
        ch_mapped.get_frontier((0, 1)), ch_sub.get_frontier((0, 1))
    )


def test_cheche_prediction_api_multiclass():
    """CheChe should expose prediction helpers mirroring ShuShu."""

    X0 = np.array(
        [
            [0.2, 0.2],
            [0.2, 0.8],
            [0.8, 0.8],
            [0.8, 0.2],
            [0.5, 0.5],
        ]
    )
    X1 = X0 + 2.0
    X_train = np.vstack([X0, X1])
    y_train = np.array([0] * len(X0) + [1] * len(X1))

    yhat_fp = CheChe().fit_predict(X_train, y_train)
    assert yhat_fp.shape == y_train.shape

    ch = CheChe().fit(X_train, y_train)
    X_test = np.array([[0.5, 0.5], [2.5, 2.5]])
    y_test = np.array([0, 1])

    yhat = ch.predict(X_test)
    assert np.array_equal(yhat, y_test)

    proba = ch.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    df = ch.predict_regions(X_test)
    assert np.array_equal(df["label"].to_numpy(), y_test)
    assert df["region_id"].min() >= 0

    scores = ch.decision_function(X_test)
    assert scores.shape == (len(X_test), len(ch.regions_))
    assert np.array_equal(np.argmax(scores, axis=1), df["region_id"].to_numpy())
