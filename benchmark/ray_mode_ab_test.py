import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('src')
from sheshe import ModalBoundaryClustering


def run_ab_test(n_reps: int = 3):
    sizes = [100, 400]
    results = []
    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=4, n_informative=3, n_redundant=0, random_state=0)
        t_grid = []
        t_grad = []
        acc_grid = []
        acc_grad = []
        for _ in range(n_reps):
            sh_grid = ModalBoundaryClustering(
                base_estimator=LogisticRegression(max_iter=200),
                task='classification',
                random_state=0,
                ray_mode='grid',
                scan_steps=12,
                max_subspaces=5,
            )
            start = time.perf_counter()
            sh_grid.fit(X, y)
            t_grid.append(time.perf_counter() - start)
            acc_grid.append(sh_grid.score(X, y))

            sh_grad = ModalBoundaryClustering(
                base_estimator=LogisticRegression(max_iter=200),
                task='classification',
                random_state=0,
                ray_mode='grad',
                scan_steps=12,
                max_subspaces=5,
            )
            start = time.perf_counter()
            sh_grad.fit(X, y)
            t_grad.append(time.perf_counter() - start)
            acc_grad.append(sh_grad.score(X, y))

        results.append({
            'n_samples': n,
            'time_grid_mean': float(np.mean(t_grid)),
            'time_grad_mean': float(np.mean(t_grad)),
            'speedup': float(np.mean(t_grid) / np.mean(t_grad)),
            'acc_grid_mean': float(np.mean(acc_grid)),
            'acc_grad_mean': float(np.mean(acc_grad)),
            'acc_diff_max': float(np.max(np.abs(np.array(acc_grid) - np.array(acc_grad))))
        })
    df = pd.DataFrame(results)
    df.to_csv('benchmark/ray_mode_ab_test.csv', index=False)
    return df


if __name__ == '__main__':
    df = run_ab_test()
    print(df)
