import numpy as np
import matplotlib.pyplot as plt
import pytest



from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sheshe import ModalBoundaryClustering, ShuShu, CheChe, ModalScoutEnsemble


def _make_mbc():
    return ModalBoundaryClustering(random_state=0)


def _make_shushu():
    return ShuShu(random_state=0)


def _make_cheche():
    return CheChe(random_state=0)


def _make_mse():
    return ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    )


ESTIMATORS = [
    ("mbc", _make_mbc, {}, True),
    ("shushu", _make_shushu, {}, True),
    ("cheche", _make_cheche, {"class_index": 0}, False),
    ("mse", _make_mse, {"model_idx": 0, "max_pairs": 1}, True),
]


@pytest.mark.parametrize("name,ctor,plot_kwargs,pass_y", ESTIMATORS)
def test_common_prediction_api(name, ctor, plot_kwargs, pass_y, monkeypatch):
    X, y = load_iris(return_X_y=True)
    model = ctor()
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    model2 = ctor()
    y_fp = model2.fit_predict(X, y)
    assert y_fp.shape == y.shape

    scores = model.decision_function(X)
    assert scores.shape[0] == X.shape[0]

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y)))

    df = model.predict_regions(X)
    assert list(df.columns) == ["label", "region_id"]
    assert len(df) == X.shape[0]

    called = {"flag": False}

    def fake_show():
        called["flag"] = True

    monkeypatch.setattr(plt, "show", fake_show)
    if pass_y:
        res = model.plot_pairs(X, y, **plot_kwargs)
    else:
        res = model.plot_pairs(X, **plot_kwargs)
    assert called["flag"] is False
    if res is not None:
        fig, axes = res
        from matplotlib.figure import Figure

        assert isinstance(fig, Figure)
    plt.close("all")


@pytest.mark.parametrize(
    "name,ctor", [("mbc", _make_mbc), ("shushu", _make_shushu), ("cheche", _make_cheche)]
)
def test_transform_shapes(name, ctor):
    X, y = load_iris(return_X_y=True)
    model = ctor()
    model.fit(X, y)
    T = model.transform(X)
    assert T.shape[0] == X.shape[0]
    if name in {"mbc", "cheche"}:
        assert T.shape[1] == len(model.regions_)
    else:
        assert T.shape[1] == len(model.classes_)


@pytest.mark.parametrize("ctor", [_make_mse])
def test_transform_not_implemented(ctor):
    X, y = load_iris(return_X_y=True)
    model = ctor()
    model.fit(X, y)
    with pytest.raises(NotImplementedError):
        model.transform(X)
    with pytest.raises(NotImplementedError):
        model.fit_transform(X, y)
