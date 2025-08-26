import time
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, make_classification
from sheshe.sheshe import ModalBoundaryClustering

def median_time(fun, n=3):
    vals=[]
    for _ in range(n):
        t0=time.time(); fun(); vals.append(time.time()-t0)
    return float(np.median(vals))

def test_perf_wine():
    X,y = load_wine(return_X_y=True)
    m = ModalBoundaryClustering(
        task="classification",
        random_state=42,
        base_2d_rays=8,
        scan_steps=8,
        coarse_steps=4,
        refine_steps=2,
    )
    t = median_time(lambda: m.fit(X,y), n=3)
    assert t < 1.8  # umbral laxo

def test_perf_bc_30d():
    X,y = load_breast_cancer(return_X_y=True)
    m = ModalBoundaryClustering(
        task="classification",
        random_state=42,
        base_2d_rays=8,
        scan_steps=8,
        coarse_steps=4,
        refine_steps=2,
    )
    t = median_time(lambda: m.fit(X,y), n=3)
    assert t < 5.2  # umbral laxo

def test_perf_synth_20d():
    X,y = make_classification(n_samples=800, n_features=20, n_informative=12, n_redundant=4,
                              n_classes=3, random_state=42)
    m = ModalBoundaryClustering(
        task="classification",
        random_state=42,
        base_2d_rays=8,
        scan_steps=8,
        coarse_steps=4,
        refine_steps=2,
    )
    t = median_time(lambda: m.fit(X,y), n=3)
    assert t < 1.7  # umbral laxo
