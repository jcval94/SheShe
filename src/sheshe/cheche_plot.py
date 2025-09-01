# -*- coding: utf-8 -*-
"""Plotting helpers for the :mod:`cheche` module.

These utilities visualise the 2D frontiers discovered by :class:`CheChe`.
The functions rely on :mod:`matplotlib` but import it lazily so that the
rest of the library remains usable even if plotting dependencies are
missing.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def plot_cheche(
    cheche: "CheChe",
    X: np.ndarray,
    *,
    class_index: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
):
    """Plot the frontiers discovered by ``cheche`` on top of the data ``X``.

    Parameters
    ----------
    cheche:
        A fitted :class:`CheChe` instance.
    X:
        Original data used for fitting. Only the dimensions present in the
        selected pairs are accessed.
    class_index:
        When ``cheche`` was trained in multiclass or regression mode, this
        selects which class/decile frontier to plot.
    feature_names:
        Optional names for the features. If ``None``, ``cheche.feature_names_``
        is used.

    Returns
    -------
    (fig, axes):
        A tuple containing the created :class:`matplotlib.figure.Figure` and the
        list of :class:`matplotlib.axes.Axes` objects.
    """

    import matplotlib.pyplot as plt  # local import to keep optional dependency

    if feature_names is None:
        feature_names = cheche.feature_names_

    pairs = list(cheche.pairs_)
    if not pairs:
        raise ValueError("CheChe instance has no stored pairs. Did you call fit()?")

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (i, j) in zip(axes, pairs):
        pts = X[:, [i, j]]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5, label="data")
        frontier = cheche.get_frontier((i, j), class_index=class_index)
        frontier = np.concatenate([frontier, frontier[:1]], axis=0)
        ax.plot(frontier[:, 0], frontier[:, 1], color="red", label="frontier")
        if feature_names is not None and len(feature_names) > max(i, j):
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
        ax.legend()

    fig.tight_layout()
    return fig, axes


__all__ = ["plot_cheche"]
