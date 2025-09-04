import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sheshe import CheChe


def test_plot_pairs_returns_axes():
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    ch = CheChe().fit(X)
    fig, axes = ch.plot_pairs(X)
    assert len(axes) == len(ch.pairs_)
    plt.close(fig)


def test_plot_classes_heatmap():
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    y = np.array([0, 0, 1, 1])
    ch = CheChe().fit(X, y)
    ch.plot_classes(X, y, heatmap=True, grid_res=5)
    titles = [plt.figure(n).axes[0].get_title() for n in plt.get_fignums()]
    for cid in ch.per_class_.keys():
        assert any(str(cid) in t for t in titles)
    plt.close("all")
