"""Meta-optimization utilities.

This module provides simple, gradient-free search routines that can be used
for meta-optimization.  The intent is to offer lightweight tools for
experimenting with hyperparameters without relying on first-order
approximations such as gradients or SPSA.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.utils import check_random_state


def _sample(space: Sequence[Any] | Tuple[float, float], rng: np.random.RandomState) -> Any:
    """Sample a value from a parameter ``space``.

    Parameters
    ----------
    space:
        Either a sequence of categorical options or a tuple ``(low, high)``
        defining a closed interval.  When ``low`` and ``high`` are integers the
        sampling is done using ``randint`` (inclusive).
    rng:
        Random number generator.
    """
    if isinstance(space, tuple) and len(space) == 2:
        lo, hi = space
        if isinstance(lo, int) and isinstance(hi, int):
            return int(rng.randint(lo, hi + 1))
        return float(rng.uniform(float(lo), float(hi)))
    return rng.choice(list(space))


def random_search(
    objective_fn: Callable[[Dict[str, Any]], float],
    param_distributions: Dict[str, Sequence[Any] | Tuple[float, float]],
    n_iter: int = 25,
    random_state: int | None = None,
) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
    """Random search meta-optimization.

    Parameters
    ----------
    objective_fn:
        Function that receives a parameter dictionary and returns a scalar score
        to maximise.  Larger is assumed to be better.
    param_distributions:
        Mapping from parameter name to either a sequence of options or an
        interval ``(low, high)`` from which values will be sampled uniformly.
    n_iter:
        Number of random configurations to evaluate.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    best_params:
        Parameter combination with the highest score.
    best_score:
        Score returned by ``objective_fn`` for ``best_params``.
    history:
        List of all evaluated parameter dictionaries paired with their score.
    """
    rng = check_random_state(random_state)
    history: List[Tuple[Dict[str, Any], float]] = []
    best_params: Dict[str, Any] | None = None
    best_score = -np.inf

    for _ in range(int(n_iter)):
        params = {name: _sample(space, rng) for name, space in param_distributions.items()}
        score = float(objective_fn(params))
        history.append((params, score))
        if score > best_score:
            best_score = score
            best_params = params

    assert best_params is not None  # n_iter >=1 ensures this
    return best_params, best_score, history
