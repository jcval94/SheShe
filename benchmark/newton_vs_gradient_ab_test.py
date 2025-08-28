import time
import numpy as np
import pandas as pd
import sys
sys.path.append('src')
from sheshe.sheshe import gradient_ascent, newton_trust_region


def quadratic_problem():
    def f(x):
        return -((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)

    def grad(x):
        return np.array([-2.0 * (x[0] - 1.0), -2.0 * (x[1] + 2.0)])

    def hess(x):
        return np.array([[-2.0, 0.0], [0.0, -2.0]])

    return f, grad, hess


class Counter:
    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return self.fn(x)


def run_ab_test(n_reps: int = 5, seed: int = 0):
    f, grad, hess = quadratic_problem()
    lo = np.array([-10.0, -10.0])
    hi = np.array([10.0, 10.0])
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_reps):
        x0 = rng.uniform(-9.0, 9.0, size=2)
        f_g = Counter(f)
        grad_g = Counter(grad)
        start = time.perf_counter()
        gradient_ascent(f_g, x0, (lo, hi), lr=0.2, max_iter=500, gradient=grad_g)
        t_grad = time.perf_counter() - start

        f_n = Counter(f)
        grad_n = Counter(grad)
        hess_n = Counter(hess)
        start = time.perf_counter()
        newton_trust_region(
            f_n, x0, (lo, hi), gradient=grad_n, hessian=hess_n, trust_radius=10.0
        )
        t_newton = time.perf_counter() - start

        rows.append(
            {
                'time_grad': t_grad,
                'time_newton': t_newton,
                'f_evals_grad': f_g.n,
                'grad_evals_grad': grad_g.n,
                'f_evals_newton': f_n.n,
                'grad_evals_newton': grad_n.n,
                'hess_evals_newton': hess_n.n,
            }
        )
    df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            'time_grad_mean': [df['time_grad'].mean()],
            'time_newton_mean': [df['time_newton'].mean()],
            'speedup': [df['time_grad'].mean() / df['time_newton'].mean()],
            'f_evals_grad_mean': [df['f_evals_grad'].mean()],
            'f_evals_newton_mean': [df['f_evals_newton'].mean()],
            'grad_evals_grad_mean': [df['grad_evals_grad'].mean()],
            'grad_evals_newton_mean': [df['grad_evals_newton'].mean()],
            'hess_evals_newton_mean': [df['hess_evals_newton'].mean()],
        }
    )
    summary.to_csv('benchmark/newton_vs_gradient_ab_test.csv', index=False)
    return summary


if __name__ == '__main__':
    df = run_ab_test()
    print(df)
