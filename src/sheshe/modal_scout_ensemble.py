from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Sequence, Callable
import math, time
import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.utils.multiclass import type_of_target

# Importaciones pedidas
from sheshe.subspace_scout import SubspaceScout
from sheshe import ModalBoundaryClustering


# ------------------------
# Utilidades ligeras
# ------------------------

def jaccard_overlap(a: Sequence[int], b: Sequence[int]) -> float:
  A, B = set(a), set(b)
  u = len(A | B)
  return (len(A & B) / u) if u else 0.0

def minmax01(values: Sequence[float], eps: float = 1e-12) -> np.ndarray:
  v = np.asarray(values, dtype=float)
  vmin, vmax = float(np.min(v)), float(np.max(v))
  if math.isclose(vmin, vmax):
    return np.ones_like(v)
  return (v - vmin) / (vmax - vmin + eps)

def infer_task(y) -> str:
  t = type_of_target(y)
  return "classification" if t in ("binary", "multiclass") else "regression"

def ray_count(dim: int, base_2d: int = 8, cap: int = 48) -> int:
  """2D->8 (preferencia de JC). >=3D: ceil(8 * sqrt(d/2)), con tope."""
  if dim <= 2:
    return base_2d
  return int(min(cap, math.ceil(base_2d * math.sqrt(dim / 2.0))))


def pick_slices(
  subspaces: List[Dict[str, Any]],
  *,
  top_k: int = 8,
  min_score: Optional[float] = None,
  max_order: Optional[int] = None,
  metric: Optional[str] = "mi_synergy",
  jaccard_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
  """Filtra por métrica/orden/score y desduplica por Jaccard."""
  cand = [s for s in subspaces if (metric is None or s.get("metric") == metric)]
  if max_order is not None:
    cand = [s for s in cand if s.get("order", len(s["features"])) <= max_order]
  if min_score is not None:
    cand = [s for s in cand if s.get("score", -np.inf) >= min_score]
  cand.sort(key=lambda d: d.get("score", -np.inf), reverse=True)

  chosen: List[Dict[str, Any]] = []
  for s in cand:
    feats = s["features"]
    if not any(jaccard_overlap(feats, c["features"]) >= jaccard_threshold for c in chosen):
      chosen.append(s)
      if top_k is not None and len(chosen) >= top_k:
        break
  return chosen


def extract_global_importances(
  estimator: BaseEstimator,
  X: np.ndarray,
  y: np.ndarray,
  *,
  sample_size: Optional[int] = 4096,
  random_state: Optional[int] = 0,
) -> Optional[np.ndarray]:
  """
  Ajusta un clon del estimator (posible submuestreo) y devuelve
  importancias por feature si existen (feature_importances_ o coef_).
  """
  n = X.shape[0]
  Xs, ys = X, y
  if sample_size is not None and n > sample_size:
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=sample_size, replace=False)
    Xs, ys = X[idx], y[idx]

  est = clone(estimator)
  try:
    est.fit(Xs, ys)
  except Exception:
    return None

  if hasattr(est, "feature_importances_"):
    imp = np.asarray(est.feature_importances_, dtype=float)
  elif hasattr(est, "coef_"):
    coef = np.asarray(est.coef_, dtype=float)
    if coef.ndim == 1:
      imp = np.abs(coef)
    else:
      # multi-clase: suma de magnitudes por feature
      imp = np.sum(np.abs(coef), axis=0)
  else:
    return None

  if imp.ndim != 1:
    return None
  return imp


# ------------------------
# Ensamble principal
# ------------------------

class ModalScoutEnsemble(BaseEstimator):
  """
  Ensamble de ModalBoundaryClustering aplicado SOLO en subespacios valiosos
  hallados por SubspaceScout (ejecutado internamente).

  Ponderación por subespacio s (normalizado en [0,1]):
    weight_s ∝ (scout_score_s)^alpha × (cv_score_s)^beta × (feat_importance_s)^gamma

  Donde:
    • scout_score_s = score del SubspaceScout para el subespacio s
    • cv_score_s    = desempeño local del MBC en s (CV u holdout)
    • feat_importance_s = suma (o media) de importancias globales de las features en s

  Parametrización de SubspaceScout:
    Usa 'scout_kwargs' con la firma:
      SubspaceScout(
        model_method=None, max_order=3, n_bins=8, top_m=20, branch_per_parent=5,
        density_occup_min=0.03, min_support=30, sample_size=4096, task='classification',
        random_state=0, base_pairs_limit=12, beam_width=12, extend_candidate_pool=16,
        marginal_gain_min=0.001, max_eval_per_order=1000, time_budget_s=None,
        objective='mi_joint', min_per_order=1
      )
  """

  def __init__(
    self,
    *,
    base_estimator,
    task: Optional[str] = None,

    # Selección de subespacios
    top_k: int = 8,
    min_score: Optional[float] = None,
    max_order: Optional[int] = None,
    metric: Optional[str] = "mi_synergy",
    jaccard_threshold: float = 0.55,

    # Ponderación (exponentes)
    alpha: float = 0.5,  # scout
    beta: float = 0.5,   # cv
    gamma: float = 0.5,  # feature_importances (si disponibles)

    # Evaluación local (rápida)
    cv: int | None = 3,                 # 0/None => holdout 80/20
    cv_metric_cls: Callable = balanced_accuracy_score,
    cv_metric_reg: Callable = r2_score,
    cv_floor: Optional[float] = None,   # salta subespacios con CV < floor

    # Rendimiento
    n_jobs: int = 1,
    random_state: Optional[int] = 0,
    base_2d_rays: int = 8,
    ray_cap: int = 48,
    time_budget_s: Optional[float] = None,

    # Importancias globales del base_estimator
    use_importances: bool = True,
    importance_sample_size: Optional[int] = 4096,  # para entrenar imp más rápido

    # Config de SubspaceScout
    scout_kwargs: Optional[Dict[str, Any]] = None,

    # Passthrough a MBC
    mbc_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    prediction_within_region: bool = False,
  ):
    self.base_estimator = base_estimator
    self.task = task

    self.top_k = top_k
    self.min_score = min_score
    self.max_order = max_order
    self.metric = metric
    self.jaccard_threshold = jaccard_threshold

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

    self.cv = cv
    self.cv_metric_cls = cv_metric_cls
    self.cv_metric_reg = cv_metric_reg
    self.cv_floor = cv_floor

    self.n_jobs = n_jobs
    self.random_state = random_state
    self.base_2d_rays = base_2d_rays
    self.ray_cap = ray_cap
    self.time_budget_s = time_budget_s

    self.use_importances = use_importances
    self.importance_sample_size = importance_sample_size

    self.scout_kwargs = scout_kwargs or {}
    self.mbc_kwargs = mbc_kwargs or {}
    self.verbose = verbose
    self.prediction_within_region = prediction_within_region

    # Atributos post-fit
    self.selected_: List[Dict[str, Any]] = []
    self.models_: List[ModalBoundaryClustering] = []
    self.features_: List[Tuple[int, ...]] = []
    self.weights_: Optional[np.ndarray] = None
    self.cv_scores_: List[float] = []
    self.scout_scores_: List[float] = []
    self.imp_scores_: List[float] = []
    self.classes_: Optional[np.ndarray] = None
    self.fitted_task_: Optional[str] = None

  # ---------- internos ----------

  def _mk_mbc_ctor(self, dim: int):
    rays = ray_count(dim, base_2d=self.base_2d_rays, cap=self.ray_cap)
    def ctor():
      return ModalBoundaryClustering(
        base_estimator=clone(self.base_estimator),
        task=self.fitted_task_,
        base_2d_rays=rays,
        random_state=self.random_state,
        **self.mbc_kwargs
      )
    return ctor

  def _precompute_splits(self, y: np.ndarray, task: str):
    """Splits compartidos para ahorrar overhead en CV."""
    if not self.cv or self.cv <= 0:
      return None
    if task == "classification":
      cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
      return list(cv.split(np.zeros((len(y), 1)), y))
    else:
      cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
      return list(cv.split(np.zeros((len(y), 1)), y))

  def _cv_or_holdout_score(self, make_model, Xs: np.ndarray, y: np.ndarray, task: str, splits):
    if splits is None:
      # Holdout 80/20
      if task == "classification":
        X_tr, X_te, y_tr, y_te = train_test_split(
          Xs, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        m = make_model()
        m.fit(X_tr, y_tr)
        return float(self.cv_metric_cls(y_te, m.predict(X_te)))
      else:
        X_tr, X_te, y_tr, y_te = train_test_split(
          Xs, y, test_size=0.2, random_state=self.random_state
        )
        m = make_model()
        m.fit(X_tr, y_tr)
        return float(self.cv_metric_reg(y_te, m.predict(X_te)))

    # CV k-fold con splits precomputados
    scores = []
    if self.fitted_task_ == "classification":
      scorer = self.cv_metric_cls
    else:
      scorer = self.cv_metric_reg

    for tr, te in splits:
      m = make_model()
      m.fit(Xs[tr], y[tr])
      scores.append(float(scorer(y[te], m.predict(Xs[te]))))
    return float(np.mean(scores)) if scores else 0.0

  # ---------- API sklearn ----------

  def fit(self, X: np.ndarray, y: np.ndarray):
    t0 = time.time()
    self.fitted_task_ = self.task or infer_task(y)
    if self.fitted_task_ not in ("classification", "regression"):
      raise ValueError(f"Tarea no soportada: {self.fitted_task_}")

    # 1) (Opcional) Importancias globales del base_estimator
    global_imp = None
    if self.use_importances:
      global_imp = extract_global_importances(
        self.base_estimator, X, y,
        sample_size=self.importance_sample_size,
        random_state=self.random_state,
      )
      if global_imp is not None and len(global_imp) != X.shape[1]:
        # seguridad
        global_imp = None

    # 2) Ejecuta SubspaceScout INTERNAMENTE
    scout_args = dict(
      task=self.fitted_task_,
      random_state=self.random_state,
    )
    # Defaults prácticos: mantener lo pedido, pero deja override por scout_kwargs
    # Si no especificas 'objective', usamos 'mi_synergy' para alinear tu métrica
    scout_args.setdefault("objective", "mi_synergy")
    scout_args.update(self.scout_kwargs or {})

    scout = SubspaceScout(**scout_args)
    subspaces = scout.fit(X, y)  # lista de dicts con keys: features, score, metric, order, ...

    # 3) Selección + desduplicación
    self.selected_ = pick_slices(
      subspaces,
      top_k=self.top_k,
      min_score=self.min_score,
      max_order=self.max_order,
      metric=self.metric,
      jaccard_threshold=self.jaccard_threshold,
    )
    if not self.selected_:
      raise ValueError("SubspaceScout no produjo subespacios útiles tras el filtrado.")

    self.models_.clear()
    self.features_.clear()
    self.cv_scores_.clear()
    self.scout_scores_ = [float(s.get("score", 0.0)) for s in self.selected_]
    self.imp_scores_.clear()

    # 4) Precompute CV splits
    splits = self._precompute_splits(y, self.fitted_task_)

    # 5) Entrenamiento por subespacio (paralelizable)
    use_joblib = (self.n_jobs is not None and self.n_jobs != 1)
    if use_joblib:
      try:
        from joblib import Parallel, delayed
      except Exception:
        use_joblib = False

    def _train_one(s: Dict[str, Any]):
      feats = tuple(s["features"])
      Xs = X[:, feats]  # vista sin copia
      ctor = self._mk_mbc_ctor(dim=len(feats))

      # CV / Holdout local
      cv_s = self._cv_or_holdout_score(ctor, Xs, y, self.fitted_task_, splits)

      # early-skip por piso
      if self.cv_floor is not None and cv_s < self.cv_floor:
        return feats, None, cv_s, 0.0

      # Importancia del subespacio (si existe)
      imp_s = 0.0
      if global_imp is not None:
        # suma de importancias de las features del subespacio
        imp_s = float(np.sum(np.abs(global_imp[list(feats)])))

      # Ajuste final
      model = ctor()
      model.fit(Xs, y)
      return feats, model, cv_s, imp_s

    if use_joblib:
      from joblib import Parallel, delayed
      results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
        delayed(_train_one)(s) for s in self.selected_
      )
    else:
      results = []
      for s in self.selected_:
        if self.time_budget_s is not None and (time.time() - t0) >= self.time_budget_s:
          if self.verbose:
            print("[ModalScoutEnsemble] Budget de tiempo alcanzado; se detiene.")
          break
        results.append(_train_one(s))

    # 6) Consolidación
    kept_scout_scores, kept_cv_scores, kept_imp_scores = [], [], []
    for (feats, model, cv_s, imp_s), sc_score in zip(results, self.scout_scores_):
      self.features_.append(feats)
      self.cv_scores_.append(float(cv_s))
      self.imp_scores_.append(float(imp_s))
      if model is not None:
        self.models_.append(model)
        kept_scout_scores.append(sc_score)
        kept_cv_scores.append(float(cv_s))
        kept_imp_scores.append(float(imp_s))

    if not self.models_:
      raise RuntimeError("No se entrenó ningún MBC (p. ej., por cv_floor o budget).")

    # 7) Pesos
    scout_norm = minmax01(kept_scout_scores) if len(kept_scout_scores) else np.ones(len(self.models_))
    cv_norm    = minmax01(kept_cv_scores)    if len(kept_cv_scores)    else np.ones(len(self.models_))

    if np.all(np.asarray(kept_imp_scores) == 0.0) or not self.use_importances:
      imp_norm = np.ones(len(self.models_))
      g = 0.0  # no aportan
    else:
      imp_norm = minmax01(kept_imp_scores)
      g = float(self.gamma)

    a = float(self.alpha)
    b = float(self.beta)
    raw = (scout_norm ** a) * (cv_norm ** b) * (imp_norm ** g)
    denom = float(np.sum(raw)) or 1.0
    self.weights_ = (raw / denom).astype(float)

    # 8) Clases (si clasificación)
    if self.fitted_task_ == "classification":
      c = None
      for m in self.models_:
        if hasattr(m, "classes_"):
          c = np.asarray(m.classes_)
          break
      if c is None:
        c = np.unique(y)
      self.classes_ = c

    # 9) Assign global cluster IDs across all submodels
    self.regions_ = []
    cid = 0
    for mbc in self.models_:
      if hasattr(mbc, "regions_"):
        for reg in mbc.regions_:
          reg.cluster_id = cid
          self.regions_.append(reg)
          cid += 1

    if self.prediction_within_region:
      self.labels_, self.label2id_ = self.predict_regions(X)
    else:
      self.labels_ = self.predict(X)

    if self.verbose:
      print(f"[ModalScoutEnsemble] Submodelos={len(self.models_)} | Pesos≈{np.round(self.weights_, 3)}")
    return self

  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    if self.fitted_task_ != "classification":
      raise AttributeError("predict_proba solo para clasificación.")
    if self.classes_ is None:
      raise RuntimeError("Modelo no ajustado.")

    n, k = X.shape[0], len(self.classes_)
    agg = np.zeros((n, k), dtype=float)

    for w, feats, mbc in zip(self.weights_, self.features_, self.models_):
      Xs = X[:, feats]
      if hasattr(mbc, "predict_proba"):
        P = mbc.predict_proba(Xs)
        if P.shape[1] != k:
          subc = getattr(mbc, "classes_", self.classes_)
          P2 = np.zeros((n, k), dtype=float)
          for j, c in enumerate(subc):
            idx = int(np.where(self.classes_ == c)[0][0])
            P2[:, idx] = P[:, j]
          P = P2
      else:
        yhat = mbc.predict(Xs)
        P = np.zeros((n, k), dtype=float)
        for j, c in enumerate(self.classes_):
          P[:, j] = (yhat == c).astype(float)
      agg += w * P

    row_sums = agg.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return agg / row_sums

  def predict(self, X: np.ndarray) -> np.ndarray:
    if self.fitted_task_ == "classification":
      P = self.predict_proba(X)
      return self.classes_[np.argmax(P, axis=1)]
    # Regresión: promedio ponderado
    out = None
    for w, feats, mbc in zip(self.weights_, self.features_, self.models_):
      Xs = X[:, feats]
      yhat = mbc.predict(Xs)
      out = (w * yhat) if out is None else (out + w * yhat)
    return out

  def predict_regions(self, X: np.ndarray):
    """Predicción usando solo modelos cuya región cubre cada muestra.

    Devuelve una tupla ``(labels, cluster_ids)``. Para cada muestra:

    * ``labels[i]`` es la predicción agregada usando únicamente los submodelos
      que incluyen a ``X[i]`` dentro de alguna de sus regiones. Si ningún
      submodelo cubre la muestra, el valor es ``-1``.
    * ``cluster_ids[i]`` es el ``cluster_id`` del submodelo con mayor peso que
      cubre la muestra. Si ninguno la cubre, toma el valor ``-1``.
    """

    if self.weights_ is None:
      raise RuntimeError("Modelo no ajustado.")

    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if self.fitted_task_ == "classification":
      labels = np.empty(n, dtype=object)
    else:
      labels = np.full(n, -1.0, dtype=float)
    cluster_ids = np.full(n, -1, dtype=int)

    for i in range(n):
      agg = None
      used_weights = []
      used_ids = []
      for w, feats, mbc in zip(self.weights_, self.features_, self.models_):
        Xi = X[i : i + 1, feats]
        cid = mbc.predict_regions(Xi)[0]
        cid_val = cid[0] if isinstance(cid, list) else cid
        if cid_val == -1:
          continue
        if self.fitted_task_ == "classification":
          if hasattr(mbc, "predict_proba"):
            P = mbc.predict_proba(Xi)
            if P.shape[1] != len(self.classes_):
              subc = getattr(mbc, "classes_", self.classes_)
              P2 = np.zeros((1, len(self.classes_)), dtype=float)
              for j, c in enumerate(subc):
                idx = int(np.where(self.classes_ == c)[0][0])
                P2[0, idx] = P[0, j]
              P = P2
          else:
            yhat = mbc.predict(Xi)
            P = np.zeros((1, len(self.classes_)), dtype=float)
            for j, c in enumerate(self.classes_):
              P[0, j] = float(yhat[0] == c)
          agg = w * P if agg is None else agg + w * P
        else:
          yhat = mbc.predict(Xi)
          agg = w * yhat if agg is None else agg + w * yhat
        used_weights.append(w)
        used_ids.append(cid_val)

      if used_weights:
        wsum = sum(used_weights)
        if self.fitted_task_ == "classification":
          probs = agg / wsum
          labels[i] = self.classes_[int(np.argmax(probs))]
        else:
          labels[i] = float(agg / wsum)
        cluster_ids[i] = used_ids[int(np.argmax(used_weights))]
      else:
        labels[i] = -1
        cluster_ids[i] = -1

    return labels, cluster_ids

  def report(self) -> List[Dict[str, Any]]:
    """Resumen por subespacio (ordenado por peso)."""
    info = []
    # Solo reportamos para los modelos realmente entrenados
    for idx, (w, feats, mbc) in enumerate(zip(self.weights_, self.features_, self.models_)):
      s = self.selected_[idx] if idx < len(self.selected_) else {}
      row = {
        "features": feats,
        "order": s.get("order", len(feats)),
        "metric": s.get("metric"),
        "scout_score": float(s.get("score", 0.0)),
        "cv_score": float(self.cv_scores_[idx]) if idx < len(self.cv_scores_) else None,
        "feat_importance": float(self.imp_scores_[idx]) if idx < len(self.imp_scores_) else None,
        "weight": float(w),
        "cluster_ids": [reg.cluster_id for reg in getattr(mbc, "regions_", [])],
      }
      for attr in ("regions_", "rules_", "segments_", "boundaries_", "feature_importances_"):
        if hasattr(mbc, attr):
          row[attr] = getattr(mbc, attr)
      info.append(row)
    info.sort(key=lambda d: d["weight"], reverse=True)
    return info
