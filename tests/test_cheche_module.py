import numpy as np

from sheshe import CheChe


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


def test_cheche_multiclass_frontiers():
    # two classes forming disjoint squares in the (0,1) plane
    X0 = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
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
