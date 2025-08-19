from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


# -----------------------------
# 1) Caché de scores (proba/margen)
# -----------------------------
class _ScoreCache:
    """Caché simple con opción memmap para N muy grande."""
    def __init__(self, use_cache: bool = True, dtype: str = "float32",
                 use_memmap: bool = False, memmap_dir: Optional[str] = None):
        self.use_cache = use_cache
        self.dtype = np.float32 if dtype == "float32" else np.float64
        self.use_memmap = use_memmap
        self.memmap_dir = memmap_dir
        self._arr = None
        self._key = None
        self._memmap_path = None

    def _new_array(self, shape):
        if not self.use_memmap:
            return np.empty(shape, dtype=self.dtype)
        if self.memmap_dir is None:
            self.memmap_dir = tempfile.gettempdir()
        fd, path = tempfile.mkstemp(prefix="mbc_scores_", dir=self.memmap_dir)
        os.close(fd)
        self._memmap_path = path
        return np.memmap(path, mode="w+", dtype=self.dtype, shape=shape)

    def clear(self):
        self._arr = None
        self._key = None
        if self._memmap_path and os.path.exists(self._memmap_path):
            try:
                os.remove(self._memmap_path)
            except Exception:
                pass
        self._memmap_path = None

    def get(self, key):
        if not self.use_cache:
            return None
        if self._key == key and self._arr is not None:
            return self._arr
        return None

    def set(self, key, values: np.ndarray):
        if not self.use_cache:
            return
        out = self._new_array(values.shape)
        np.copyto(out, values.astype(self.dtype, copy=False))
        self._arr = out
        self._key = key


# -----------------------------------------
# 2) Submuestreo estratificado / aleatorio
# -----------------------------------------
def _stratified_indices(y: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    classes, y_idx = np.unique(y, return_inverse=True)
    # reparte proporcionalmente
    counts = np.bincount(y_idx)
    props = counts / counts.sum()
    take = np.maximum(1, np.floor(props * sample_size)).astype(int)
    idxs = []
    for c in range(len(classes)):
        ids = np.flatnonzero(y_idx == c)
        k = min(len(ids), take[c])
        idxs.append(rng.choice(ids, size=k, replace=False))
    out = np.concatenate(idxs)
    if out.size > sample_size:  # recorte fino si por redondeos te pasas
        out = rng.choice(out, size=sample_size, replace=False)
    return np.sort(out)

def _random_indices(n: int, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    sample_size = min(n, sample_size)
    return np.sort(rng.choice(n, size=sample_size, replace=False))


# ---------------------------------------------------
# 3) Percentil rápido con auto-heurística para big-N
# ---------------------------------------------------
def percentile_threshold_auto(values: np.ndarray, q: float,
                              y: Optional[np.ndarray] = None,
                              sample_size: Optional[int] = 50000,
                              method: str = "auto",
                              hist_bins: int = 2048,
                              rng_seed: int = 0) -> float:
    """
    Estima el percentil q de 'values' con:
      - submuestreo (opcional) para datasets grandes
      - método 'partition' por defecto
      - hist-sketch si el submuestreo sigue siendo enorme

    q: 0..1 (e.g., 0.10 para percentil 10)
    sample_size: si None o >= len(values), usa todo el vector.
    method: 'auto' | 'partition' | 'hist'
    """
    v = np.asarray(values)
    n = v.size
    if n == 0:
        raise ValueError("values vacío")

    rng = np.random.default_rng(rng_seed)

    # 1) Submuestreo opcional
    if sample_size is not None and sample_size < n:
        if y is not None and y.ndim == 1 and np.issubdtype(y.dtype, np.integer):
            idx = _stratified_indices(y, sample_size, rng)
        else:
            idx = _random_indices(n, sample_size, rng)
        v = v[idx]
        n = v.size

    # Sanitiza y decide método
    v = v[np.isfinite(v)]
    if n == 0:
        raise ValueError("values no finito")

    if method == "auto":
        method = "partition" if n <= 2_000_000 else "hist"

    if method == "partition":
        k = int(np.floor(q * (n - 1)))
        tmp = v.copy()
        return np.partition(tmp, k)[k]

    if method == "hist":
        vmin, vmax = float(v.min()), float(v.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return vmin
        counts, edges = np.histogram(v, bins=hist_bins, range=(vmin, vmax))
        cum = np.cumsum(counts)
        target = q * n
        bin_idx = int(np.searchsorted(cum, target, side="left"))
        bin_idx = int(np.clip(bin_idx, 0, len(edges) - 2))
        left, right = edges[bin_idx], edges[bin_idx + 1]
        prev = cum[bin_idx - 1] if bin_idx > 0 else 0
        within = (target - prev) / max(1, counts[bin_idx])
        return float(left + (right - left) * within)

    raise ValueError(f"Método desconocido: {method}")


# -----------------------------------------
# 4) Vectorizar el "percentile drop" check
# -----------------------------------------
def mask_above_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Devuelve una máscara booleana (scores >= threshold), vectorizada."""
    return np.asarray(scores) >= threshold


# -----------------------------------------
# 5) Helper: invalidación de caché
# -----------------------------------------
def _fingerprint_model(model) -> Tuple:
    """Huella ligera del modelo para invalidar caché (ajústala a tu caso)."""
    attrs = []
    for name in ("__class__", "n_estimators", "max_depth", "C", "alpha", "random_state"):
        val = getattr(model, name, None)
        attrs.append(val if (name != "__class__") else str(val))
        
    return tuple(attrs)
