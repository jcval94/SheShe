import cProfile, pstats, io, time
from sklearn.datasets import load_iris
from sheshe.sheshe import ModalBoundaryClustering

X, y = load_iris(return_X_y=True)
m = ModalBoundaryClustering(task="classification", random_state=42)

pr = cProfile.Profile()
t0 = time.time(); pr.enable(); m.fit(X, y); pr.disable(); t = time.time() - t0
s = io.StringIO(); pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(30)
print(f"fit median ~ {t:.3f}s"); print(s.getvalue())
