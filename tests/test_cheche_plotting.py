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
