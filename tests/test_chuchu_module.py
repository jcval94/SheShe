import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import load_iris

from sheshe import ChuchuClassifier, ChuchuRegressor, ChuchuConfig
from sheshe.chuchu import macro_f1_ignore_rejects


def test_chuchu_classifier_basic():
    data = load_iris()
    X, y = data.data, data.target
    clf = ChuchuClassifier(ChuchuConfig()).fit(X, y)
    yhat = clf.predict(X)
    assert yhat.shape == y.shape
    proba = clf.predict_proba(X[:5])
    assert set(proba.keys()) == set(np.unique(y))
    score = macro_f1_ignore_rejects(y, yhat)
    assert 0.5 < score <= 1.0


def test_chuchu_regressor_basic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = 0.5 * X[:, 0] + 0.2 * X[:, 1]
    reg = ChuchuRegressor(ChuchuConfig()).fit(X, y)
    preds = reg.predict(X[:10])
    assert preds.shape == (10,)
    mask = reg.region_mask(X[:10])
    assert mask.shape == (10,)
    assert np.isfinite(preds).all()


def test_chuchu_classifier_utils(tmp_path):
    data = load_iris()
    X, y = data.data, data.target
    clf = ChuchuClassifier(ChuchuConfig())
    y_fp = clf.fit_predict(X, y)
    clf2 = ChuchuClassifier(ChuchuConfig()).fit(X, y)
    y_seq = clf2.predict(X)
    np.testing.assert_array_equal(y_fp, y_seq)
    dec = clf2.decision_function(X[:5])
    assert dec.shape == (5, len(np.unique(y)))
    trans = clf2.transform(X[:5])
    np.testing.assert_allclose(dec, trans)
    clf3 = ChuchuClassifier(ChuchuConfig())
    ft = clf3.fit_transform(X, y)
    np.testing.assert_allclose(ft, clf3.transform(X))
    path = tmp_path / "clf.joblib"
    clf2.save(path)
    loaded = ChuchuClassifier.load(path)
    np.testing.assert_array_equal(clf2.predict(X), loaded.predict(X))


def test_chuchu_regressor_utils(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = 0.5 * X[:, 0] + 0.2 * X[:, 1]
    reg = ChuchuRegressor(ChuchuConfig())
    pred_fp = reg.fit_predict(X, y)
    reg2 = ChuchuRegressor(ChuchuConfig()).fit(X, y)
    pred_seq = reg2.predict(X)
    np.testing.assert_allclose(pred_fp, pred_seq)
    dec = reg2.decision_function(X[:5])
    assert dec.shape == (5,)
    trans = reg2.transform(X[:5])
    np.testing.assert_allclose(dec, trans)
    reg3 = ChuchuRegressor(ChuchuConfig())
    ft = reg3.fit_transform(X, y)
    np.testing.assert_allclose(ft, reg3.transform(X))
    path = tmp_path / "reg.joblib"
    reg2.save(path)
    loaded = ChuchuRegressor.load(path)
    np.testing.assert_allclose(reg2.predict(X[:10]), loaded.predict(X[:10]))


def test_predict_regions_classifier():
    data = load_iris()
    X, y = data.data, data.target
    clf = ChuchuClassifier(ChuchuConfig()).fit(X, y)
    df = clf.predict_regions(X[:5])
    assert list(df.columns) == ["label", "region_id"]
    yhat = clf.predict(X[:5])
    np.testing.assert_array_equal(df["label"].to_numpy(), yhat)
    mem = clf.membership(X[:5])
    mask_from_df = df["region_id"].to_numpy() >= 0
    expected = np.array([mem[yhat[i]][i] if yhat[i] != -1 else False for i in range(len(yhat))])
    np.testing.assert_array_equal(mask_from_df, expected)


def test_predict_regions_regressor():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = 0.5 * X[:, 0] + 0.2 * X[:, 1]
    reg = ChuchuRegressor(ChuchuConfig()).fit(X, y)
    df = reg.predict_regions(X[:5])
    assert list(df.columns) == ["label", "region_id"]
    preds = reg.predict(X[:5])
    np.testing.assert_allclose(df["label"].to_numpy(), preds)
    mask = reg.region_mask(X[:5])
    np.testing.assert_array_equal(df["region_id"].to_numpy() >= 0, mask)


def test_chuchu_plotting():
    data = load_iris()
    X, y = data.data[:, :2], data.target
    clf = ChuchuClassifier(ChuchuConfig()).fit(X, y)
    fig, axes = clf.plot_classes(X, y)
    assert len(axes) == 1
    plt.close(fig)
    figp, axesp = clf.plot_pairs(X)
    assert len(axesp) == 1
    plt.close(figp)
    with pytest.raises(NotImplementedError):
        clf.plot_pair_3d(X, (0, 1))
    target = 0.5 * X[:, 0] + 0.2 * X[:, 1]
    reg = ChuchuRegressor(ChuchuConfig()).fit(X, target)
    fig2, axes2 = reg.plot_classes(X, target)
    assert len(axes2) == 1
    plt.close(fig2)
    figp2, axesp2 = reg.plot_pairs(X)
    assert len(axesp2) == 1
    plt.close(figp2)
    with pytest.raises(NotImplementedError):
        reg.plot_pair_3d(X, (0, 1))
