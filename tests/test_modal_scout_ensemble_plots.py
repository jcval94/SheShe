import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sheshe import ModalScoutEnsemble


def _fit_mse():
    X, y = load_iris(return_X_y=True)
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    ).fit(X, y)
    return mse, X, y


def test_plot_pairs_runs():
    mse, X, y = _fit_mse()
    mse.plot_pairs(X, y, model_idx=0, max_pairs=1)


def test_plot_pair_3d_runs():
    mse, X, y = _fit_mse()
    feats = mse.features_[0]
    pair = (feats[0], feats[1])
    mse.plot_pair_3d(X, pair, model_idx=0, class_label=mse.classes_[0])


def test_mse_feature_names_propagated():
    mse, X, y = _fit_mse()
    feats = mse.features_[0]
    names = [f'f{i}' for i in range(X.shape[1])]
    mse.plot_pairs(X, y, model_idx=0, max_pairs=1, feature_names=names)
    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == names[feats[0]]
    assert ax.get_ylabel() == names[feats[1]]
    plt.close('all')

    pair = (feats[0], feats[1])
    fig = mse.plot_pair_3d(X, pair, model_idx=0, class_label=mse.classes_[0], feature_names=names)
    ax3d = fig.axes[0]
    assert ax3d.get_xlabel() == names[feats[0]]
    assert ax3d.get_ylabel() == names[feats[1]]
    plt.close('all')
