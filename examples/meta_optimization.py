"""Meta-optimization example using gradient-free random search.

This example tunes a few ``ModalBoundaryClustering`` hyperparameters on the
Iris dataset.  The search routine relies on direct evaluation of random
configurations, avoiding the first-order approximations used in other parts of
this project.
"""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sheshe import ModalBoundaryClustering
from sheshe.meta_optimization import random_search


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)

    def objective(params):
        model = ModalBoundaryClustering(
            task="classification",
            base_2d_rays=int(params["base_2d_rays"]),
            scan_steps=int(params["scan_steps"]),
            drop_fraction=float(params["drop_fraction"]),
            random_state=0,
        ).fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    space = {
        "base_2d_rays": (6, 20),
        "scan_steps": (8, 30),
        "drop_fraction": (0.3, 0.7),
    }

    best_params, best_score, history = random_search(
        objective, space, n_iter=20, random_state=0
    )
    print("Best params:", best_params)
    print("Validation accuracy:", best_score)


if __name__ == "__main__":
    main()
