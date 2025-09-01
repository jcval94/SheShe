import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sheshe import CheChe, plot_cheche


def test_plot_cheche_returns_axes():
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    ch = CheChe().fit(X)
    fig, axes = plot_cheche(ch, X)
    assert len(axes) == len(ch.pairs_)
    plt.close(fig)
