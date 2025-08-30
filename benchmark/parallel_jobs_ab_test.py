import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('src')
from sheshe import ModalScoutEnsemble

warnings.filterwarnings("ignore")

def run_ab_test(n_reps: int = 3):
    def _breast_cancer():
        data = load_breast_cancer()
        kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        X = kb.fit_transform(data.data)
        return X, data.target

    datasets = [
        ("breast_cancer", _breast_cancer),
        ("digits", lambda: load_digits(return_X_y=True)),
    ]
    submodels_list = [2, 4]
    scan_steps = 24
    results = []
    for dataset_name, loader in datasets:
        X, y = loader()
        n_samples = X.shape[0]
        for n_submodels in submodels_list:
            t_seq = []
            t_par = []
            for _ in range(n_reps):
                model_seq = ModalScoutEnsemble(
                    base_estimator=LogisticRegression(max_iter=100),
                    task='classification',
                    top_k=n_submodels,
                    n_jobs=1,
                    cv=0,
                    use_importances=False,
                    mbc_kwargs={'scan_steps': scan_steps, 'n_max_seeds': 1},
                    random_state=0,
                )
                start = time.perf_counter()
                model_seq.fit(X, y)
                t_seq.append(time.perf_counter() - start)

                model_par = ModalScoutEnsemble(
                    base_estimator=LogisticRegression(max_iter=100),
                    task='classification',
                    top_k=n_submodels,
                    n_jobs=-1,
                    cv=0,
                    use_importances=False,
                    mbc_kwargs={'scan_steps': scan_steps, 'n_max_seeds': 1},
                    random_state=0,
                )
                start = time.perf_counter()
                model_par.fit(X, y)
                t_par.append(time.perf_counter() - start)
            results.append({
                'dataset': dataset_name,
                'n_samples': n_samples,
                'n_submodels': n_submodels,
                'scan_steps': scan_steps,
                'time_seq_mean': float(np.mean(t_seq)),
                'time_par_mean': float(np.mean(t_par)),
                'speedup': float(np.mean(t_seq) / np.mean(t_par)),
            })
    df = pd.DataFrame(results)
    df.to_csv('benchmark/parallel_jobs_ab_test.csv', index=False)
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parallel jobs A/B test')
    parser.add_argument('--n-reps', type=int, default=3, help='Number of repetitions')
    args = parser.parse_args()
    print(run_ab_test(n_reps=args.n_reps))
