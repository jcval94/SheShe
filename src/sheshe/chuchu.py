# chuchu.py
# =============================================================================
# CHUCHU: Conformal HDR & MVS Regions for Multiclass Classification and Regression
# - N-dimensional
# - Per-class regions (Iris-like multiclass)
# - Density models: KDE, GMM (and hook for Normalizing Flows)
# - Accepts arbitrary fitted score models (LogReg, RandomForest, etc.)
# - Conformal calibration of HDR thresholds (per class)
# - Minimum-Volume Set (MVS) under coverage constraint (Monte Carlo volume)
# - Hyperparameter optimization: F1 (classification) / MAPE (regression)
# Author: ChatGPT for JC — MIT License
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib

_ARTIFACT_VERSION = "1.0"

# ---------------------------------------------------------------------
# Utils numéricos
# ---------------------------------------------------------------------
_EPS = 1e-12


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = _EPS) -> np.ndarray:
    return a / np.maximum(np.abs(b), eps)


def _whitener_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (mu, W) tal que z = (x - mu) @ W^T blanquea la covarianza."""
    X = np.asarray(X, float)
    mu = X.mean(0)
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-8)
    W = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return mu, W


def _apply_whiten(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (np.asarray(X, float) - mu) @ W.T


def _scott(n: int, d: int) -> float:
    return float(np.power(n, -1.0 / (d + 4)))


def _bounding_box(X: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    lo = np.quantile(X, q_low, axis=0)
    hi = np.quantile(X, q_high, axis=0)
    hi = np.where(hi - lo < 1e-8, lo + 1e-8, hi)
    return lo, hi


def _train_calib_split(
    X: np.ndarray, y: Optional[np.ndarray], calib_frac: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray, None]:
    """Split simple para conformal (estratificado si y está disponible)."""
    X = np.asarray(X)
    n = len(X)
    idx = np.arange(n)
    if y is None:
        rng.shuffle(idx)
        m = max(5, int(calib_frac * n))
        return idx[m:], idx[:m], None
    # estratificado por clase
    tr_list, ca_list = [], []
    for cls in np.unique(y):
        ids = idx[y == cls]
        rng.shuffle(ids)
        m = max(2, int(calib_frac * len(ids)))
        ca_list.append(ids[:m])
        tr_list.append(ids[m:])
    return np.concatenate(tr_list), np.concatenate(ca_list), None


# ---------------------------------------------------------------------
# Interfaz de modelos de densidad/score
# ---------------------------------------------------------------------
class BaseDensityModel:
    """Interfaz mínima: fit(X), score(X) -> score positivo (densidad/confianza), sample(n)."""

    def fit(self, X: np.ndarray) -> "BaseDensityModel":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return configuration parameters for this estimator."""
        return {name: getattr(self, name) for name in self.__dict__ if not name.endswith("_")}

    def set_params(self, **params) -> "BaseDensityModel":
        """Set the estimator's parameters and return ``self``."""
        for name, value in params.items():
            setattr(self, name, value)
        return self


class KDEDensityModel(BaseDensityModel):
    def __init__(self, bandwidth_factor: float = 1.0, whiten: bool = True):
        self.bandwidth_factor = float(bandwidth_factor)
        self.whiten = bool(whiten)
        self.mu_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.kde_: Optional[KernelDensity] = None

    def fit(self, X: np.ndarray) -> "KDEDensityModel":
        X = np.asarray(X, float)
        n, d = X.shape
        if self.whiten:
            self.mu_, self.W_ = _whitener_fit(X)
            Z = _apply_whiten(X, self.mu_, self.W_)
        else:
            self.mu_, self.W_ = np.zeros(d), np.eye(d)
            Z = X
        bw = max(_scott(n, d) * self.bandwidth_factor, 1e-3)
        self.kde_ = KernelDensity(bandwidth=bw, kernel="gaussian").fit(Z)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = _apply_whiten(np.asarray(X, float), self.mu_, self.W_)
        logp = self.kde_.score_samples(Z)
        p = np.exp(logp)
        return p / (p.mean() + _EPS)

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        # sampling aproximado tomando datos + jitter ~ N(0, bw^2 I) en el espacio whitened
        if rng is None:
            rng = np.random.RandomState(0)
        assert self.kde_ is not None
        Zdata = self.kde_.tree_.data  # puntos en espacio whitened
        idx = rng.randint(0, len(Zdata), size=n)
        Zs = Zdata[idx] + rng.normal(scale=self.kde_.bandwidth, size=Zdata[idx].shape)
        Xs = Zs @ np.linalg.inv(self.W_.T) + self.mu_
        return Xs


class GMMDensityModel(BaseDensityModel):
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        whiten: bool = True,
        random_state: int = 0,
    ):
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.whiten = bool(whiten)
        self.random_state = int(random_state)

        self.mu_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.gmm_: Optional[GaussianMixture] = None

    def fit(self, X: np.ndarray) -> "GMMDensityModel":
        X = np.asarray(X, float)
        n, d = X.shape
        if self.whiten:
            self.mu_, self.W_ = _whitener_fit(X)
            Z = _apply_whiten(X, self.mu_, self.W_)
        else:
            self.mu_, self.W_ = np.zeros(d), np.eye(d)
            Z = X
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        ).fit(Z)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = _apply_whiten(np.asarray(X, float), self.mu_, self.W_)
        logp = self.gmm_.score_samples(Z)
        p = np.exp(logp)
        return p / (p.mean() + _EPS)

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.RandomState(self.random_state)
        Z, _ = self.gmm_.sample(n)
        Xs = Z @ np.linalg.inv(self.W_.T) + self.mu_
        return Xs


# -----------------------------------------------------------------------------
# (Opcional) Normalizing Flows — stub listo si instalas torch+nflows
# -----------------------------------------------------------------------------
# try:
#     import torch
#     from nflows.flows import Flow
#     from nflows.distributions import StandardNormal
#     from nflows.transforms import MaskedAffineAutoregressiveTransform
#
#     class FlowDensityModel(BaseDensityModel):
#         def __init__(self, n_layers=5, hidden_features=64, whiten=True, device=None, epochs=300, lr=1e-3):
#             self.n_layers = n_layers
#             self.hidden_features = hidden_features
#             self.whiten = bool(whiten)
#             self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#             self.epochs = epochs
#             self.lr = lr
#             self.mu_ = None; self.W_ = None; self.flow_ = None
#
#         def _build(self, d):
#             transforms = []
#             for _ in range(self.n_layers):
#                 transforms.append(MaskedAffineAutoregressiveTransform(features=d, hidden_features=self.hidden_features))
#             transform = torch.nn.Sequential(*transforms)
#             base = StandardNormal([d])
#             return Flow(transform, base).to(self.device)
#
#         def fit(self, X):
#             X = np.asarray(X, float)
#             n, d = X.shape
#             if self.whiten:
#                 self.mu_, self.W_ = _whitener_fit(X)
#                 Z = _apply_whiten(X, self.mu_, self.W_)
#             else:
#                 self.mu_, self.W_ = np.zeros(d), np.eye(d)
#                 Z = X
#             self.flow_ = self._build(d)
#             opt = torch.optim.Adam(self.flow_.parameters(), lr=self.lr)
#             Zt = torch.tensor(Z, dtype=torch.float32, device=self.device)
#             for _ in range(self.epochs):
#                 opt.zero_grad()
#                 loss = -self.flow_.log_prob(Zt).mean()
#                 loss.backward(); opt.step()
#             return self
#
#         def score(self, X):
#             Z = _apply_whiten(np.asarray(X, float), self.mu_, self.W_)
#             import torch
#             Zt = torch.tensor(Z, dtype=torch.float32, device=self.device)
#             with torch.no_grad():
#                 logp = self.flow_.log_prob(Zt).cpu().numpy()
#             p = np.exp(logp)
#             return p / (p.mean() + _EPS)
#
#         def sample(self, n, rng=None):
#             import torch
#             with torch.no_grad():
#                 # samplear del base y transformar
#                 pass
# except Exception:
#     pass


# -----------------------------------------------------------------------------
# Adaptador de modelos arbitrarios (LogReg, RandomForest, etc.)
# -----------------------------------------------------------------------------
def _softmax(z, axis=1):
    z = np.asarray(z, float)
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=axis, keepdims=True) + _EPS)


def _sigmoid(z):
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class ScoreModelAdapterConfig:
    model: Any                           # modelo ya entrenado (LR, RF, XGB, etc.)
    mode: str = "auto"                   # 'auto' | 'proba' | 'decision' | 'callable'
    score_fn: Optional[Callable] = None  # si mode='callable': fn(model, X, class_label)->(n,)
    class_label: Optional[Any] = None    # etiqueta/clase para la columna/score a extraer
    proba_is_calibrated: bool = True     # info opcional
    extra: Dict[str, Any] = None         # libre

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def set_params(self, **params) -> "ScoreModelAdapterConfig":
        for name, value in params.items():
            if name in self.__dataclass_fields__:
                setattr(self, name, value)
        return self


class ScoreDensityModel(BaseDensityModel):
    """
    Adaptador per-clase: expone .score(X) = s_c(X) a partir de un modelo entrenado.
    - predict_proba: usa la columna de la clase;
    - decision_function: usa sigmoid (binario) o softmax (multiclase);
    - callable: usa score_fn(model, X, class_label).
    """
    def __init__(self, cfg: ScoreModelAdapterConfig):
        self.cfg = cfg
        self._n_features: Optional[int] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        if deep and hasattr(self.cfg, "get_params"):
            for k, v in self.cfg.get_params(deep=deep).items():
                params[f"cfg__{k}"] = v
        return params

    def set_params(self, **params) -> "ScoreDensityModel":
        cfg_prefix = "cfg__"
        cfg_params = {k[len(cfg_prefix):]: v for k, v in params.items() if k.startswith(cfg_prefix)}
        if cfg_params and hasattr(self.cfg, "set_params"):
            self.cfg.set_params(**cfg_params)
        if "cfg" in params:
            self.cfg = params["cfg"]
        remaining = {k: v for k, v in params.items() if not k.startswith(cfg_prefix) and k != "cfg"}
        super().set_params(**remaining)
        return self

    def fit(self, X: np.ndarray) -> "ScoreDensityModel":
        self._n_features = np.asarray(X).shape[1]
        return self

    def _scores_all(self, X: np.ndarray) -> np.ndarray:
        m = self.cfg.model
        mode = self.cfg.mode
        if mode == "auto":
            if hasattr(m, "predict_proba"):
                mode = "proba"
            elif hasattr(m, "decision_function"):
                mode = "decision"
            elif self.cfg.score_fn is not None:
                mode = "callable"
            else:
                raise ValueError("No se puede obtener score del modelo: agrega predict_proba, decision_function o score_fn.")

        if mode == "proba":
            P = m.predict_proba(X)
            return np.asarray(P, float)

        if mode == "decision":
            D = np.asarray(m.decision_function(X), float)
            if D.ndim == 1:  # binario
                P1 = _sigmoid(D).reshape(-1, 1)
                P = np.c_[1.0 - P1, P1]
            else:
                P = _softmax(D, axis=1)
            return P

        if mode == "callable":
            s = self.cfg.score_fn(m, X, self.cfg.class_label)
            s = np.asarray(s, float).ravel()
            # normaliza monótonamente a [0,1] (no altera orden)
            s = (s - s.min()) / (s.max() - s.min() + _EPS)
            return s.reshape(-1, 1)

        raise ValueError(f"mode desconocido: {mode}")

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        S = self._scores_all(X)  # (n, C) o (n,1)
        c = self.cfg.class_label
        if S.ndim == 1:
            return S
        if c is None:
            return S[:, -1]
        if isinstance(c, (int, np.integer)):
            return S[:, int(c)]
        if hasattr(self.cfg.model, "classes_"):
            classes = list(self.cfg.model.classes_)
            if c in classes:
                return S[:, classes.index(c)]
        return S[:, 0]

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        raise NotImplementedError("ScoreDensityModel no implementa sample(); no requerido para HDR/Conformal.")


# ---------------------------------------------------------------------
# Umbrales conformales / HDR / MVS
# ---------------------------------------------------------------------
@dataclass
class RegionSpec:
    """Especificación de región por clase (o para X total en regresión)."""
    method: str  # 'hdr' | 'mvs'
    threshold: float  # nivel de score
    coverage_target: float  # masa/coverage objetivo
    model: BaseDensityModel  # modelo de score/densidad
    meta: Dict[str, Any]     # extras (p. ej., spec de densidad)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def set_params(self, **params) -> "RegionSpec":
        for name, value in params.items():
            if name in self.__dataclass_fields__:
                setattr(self, name, value)
        return self


def _conformal_threshold(scores_calib: np.ndarray, mass: float) -> float:
    """Umbral t tal que P(score >= t) ≈ mass en el split de calibración."""
    scores_calib = np.asarray(scores_calib, float)
    return float(np.quantile(scores_calib, 1.0 - (1.0 - mass)))


def _hdr_region(
    X_train: np.ndarray,
    X_calib: np.ndarray,
    mass: float,
    model: BaseDensityModel,
) -> Tuple[float, BaseDensityModel]:
    model.fit(X_train)
    s_cal = model.score(X_calib)
    t = _conformal_threshold(s_cal, mass)
    return t, model


def _estimate_volume_MC(
    indicator: Callable[[np.ndarray], np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    n_samples: int = 20000,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """Volumen estimado por Monte Carlo dentro del hipercubo [lo, hi]."""
    rng = rng or np.random.RandomState(0)
    d = lo.shape[0]
    U = rng.uniform(size=(n_samples, d))
    Xs = lo + U * (hi - lo)
    inside = indicator(Xs).astype(float)
    frac = inside.mean()
    volume_cub = float(np.prod(hi - lo))
    return volume_cub * float(frac)


def _mvs_region(
    X_train: np.ndarray,
    X_calib: np.ndarray,
    mass: float,
    model: BaseDensityModel,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, BaseDensityModel]:
    """
    MVS ≈ seleccionar umbral t que logre cobertura >= mass en calibración
    y minimice un estimado del volumen por MC.
    """
    model.fit(X_train)
    s_cal = model.score(X_calib)

    # Candidatos: cuantiles altos del score
    qgrid = np.quantile(s_cal, np.linspace(0.5, 0.99, 25))
    best_t, best_vol = None, np.inf

    def indicator_from_t(t):
        def f(X):
            return (model.score(X) >= t).astype(int)
        return f

    for t in qgrid:
        cov = float(np.mean(s_cal >= t))
        if cov + 1e-6 < mass:
            continue
        vol = _estimate_volume_MC(indicator_from_t(t), lo, hi, n_samples=10000, rng=rng)
        if vol < best_vol:
            best_vol, best_t = vol, float(t)

    if best_t is None:
        best_t = _conformal_threshold(s_cal, mass)

    return best_t, model


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------
def f1_binary_from_masks(y_true: np.ndarray, y_pred: np.ndarray, positive_label) -> float:
    """F1 para una clase: positive_label. Rechazos (-1) en y_pred se ignoran previamente."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    yt = (y_true == positive_label)
    yp = (y_pred == positive_label)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)


def macro_f1_ignore_rejects(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 ignorando y_pred == -1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = (y_pred != -1)
    if not np.any(mask):
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    labels = np.unique(yt)
    if len(labels) == 0:
        return 0.0
    scores = [f1_binary_from_masks(yt, yp, lab) for lab in labels]
    return float(np.mean(scores))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(_safe_div(y_true - y_pred, y_true + _EPS)))) * 100.0


# ---------------------------------------------------------------------
# Núcleo CHUCHU (Clasificador y Regressor)
# ---------------------------------------------------------------------
DensitySpec = Dict[str, Any]  # {'name':'kde'|'gmm'|'flow'|'score', <kwargs>}


def _build_density_model(spec: DensitySpec) -> BaseDensityModel:
    name = spec.get("name", "kde").lower()
    if name == "kde":
        return KDEDensityModel(
            bandwidth_factor=spec.get("bandwidth_factor", 1.0),
            whiten=spec.get("whiten", True),
        )
    elif name == "gmm":
        return GMMDensityModel(
            n_components=spec.get("n_components", 3),
            covariance_type=spec.get("covariance_type", "full"),
            reg_covar=spec.get("reg_covar", 1e-6),
            whiten=spec.get("whiten", True),
            random_state=spec.get("random_state", 0),
        )
    elif name == "flow":
        raise ImportError(
            "FlowDensityModel no está habilitado aquí. Instala torch+nflows y añade la implementación (ver stub)."
        )
    elif name == "score":
        cfg = ScoreModelAdapterConfig(
            model=spec["model"],
            mode=spec.get("mode", "auto"),
            score_fn=spec.get("score_fn", None),
            class_label=spec.get("class_label", None),
            proba_is_calibrated=spec.get("proba_is_calibrated", True),
            extra={k: v for k, v in spec.items()
                   if k not in {"name", "model", "mode", "score_fn", "class_label", "proba_is_calibrated"}}
        )
        return ScoreDensityModel(cfg)
    else:
        raise ValueError(f"Modelo de densidad/score desconocido: {name}")


@dataclass
class ChuchuConfig:
    method: str = "hdr"  # 'hdr' o 'mvs'
    coverage: float = 0.90
    calib_frac: float = 0.2
    density: DensitySpec = None  # e.g., {'name':'kde','bandwidth_factor':1.2,'whiten':True}
    random_state: int = 0
    # Para regresión:
    min_coverage_reg: float = 0.60  # evita soluciones triviales en optimización

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def set_params(self, **params) -> "ChuchuConfig":
        for name, value in params.items():
            if name in self.__dataclass_fields__:
                setattr(self, name, value)
        return self


class ChuchuClassifier:
    """
    Multiclase: una región por clase. Predicción set-valued y decisión argmax de score
    restringida a clases cuya región acepta el punto.
    """

    __artifact_version__ = _ARTIFACT_VERSION

    def __init__(self, config: ChuchuConfig):
        self.config = config
        self.regions_: Dict[Any, RegionSpec] = {}
        self.priors_: Dict[Any, float] = {}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {"config": self.config}
        if deep and hasattr(self.config, "get_params"):
            params.update(self.config.get_params(deep=deep))
        return params

    def set_params(self, **params) -> "ChuchuClassifier":
        cfg_params = {k: v for k, v in params.items() if k != "config"}
        if "config" in params:
            self.config = params["config"]
        if cfg_params:
            if hasattr(self.config, "set_params"):
                self.config.set_params(**cfg_params)
            else:
                for name, value in cfg_params.items():
                    setattr(self.config, name, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChuchuClassifier":
        cfg = self.config
        rng = np.random.RandomState(cfg.random_state)
        X = np.asarray(X, float)
        y = np.asarray(y)

        self.regions_.clear()
        self.priors_.clear()

        lo, hi = _bounding_box(X)

        for cls in np.unique(y):
            Xc = X[y == cls]
            self.priors_[cls] = float(len(Xc)) / len(X)

            tr, ca, _ = _train_calib_split(Xc, None, cfg.calib_frac, rng=rng)
            Xtr, Xca = Xc[tr], Xc[ca]

            dens_spec = (cfg.density or {"name": "kde"}).copy()
            # si es adaptador de score, fija class_label para esta clase
            if dens_spec.get("name", "kde").lower() == "score":
                dens_spec["class_label"] = cls

            model = _build_density_model(dens_spec)
            if cfg.method == "hdr":
                t, m = _hdr_region(Xtr, Xca, cfg.coverage, model)
            elif cfg.method == "mvs":
                t, m = _mvs_region(Xtr, Xca, cfg.coverage, model, lo=lo, hi=hi, rng=rng)
            else:
                raise ValueError("method debe ser 'hdr' o 'mvs'")

            self.regions_[cls] = RegionSpec(
                method=cfg.method,
                threshold=float(t),
                coverage_target=float(cfg.coverage),
                model=model,
                meta={"density": dens_spec},
            )
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_kwargs) -> np.ndarray:
        """Convenience method: fit the classifier and return predictions on ``X``."""
        self.fit(X, y, **fit_kwargs)
        return self.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw scores per class ordered by class label."""
        scores = self._scores_by_class(X)
        labels = list(self.regions_.keys())
        return np.stack([scores[c] for c in labels], axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Alias for :meth:`decision_function` providing per-class scores."""
        return self.decision_function(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_kwargs) -> np.ndarray:
        """Fit the classifier and return transformed scores on ``X``."""
        self.fit(X, y, **fit_kwargs)
        return self.transform(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist the classifier to ``filepath`` using joblib."""
        payload = {"__artifact_version__": self.__artifact_version__, "model": self}
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ChuchuClassifier":
        """Load a classifier saved with :meth:`save`."""
        payload = joblib.load(filepath)
        ver = payload.get("__artifact_version__")
        if ver != cls.__artifact_version__:
            raise ValueError(
                f"Artifact version mismatch: expected {cls.__artifact_version__}, got {ver}"
            )
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("Loaded object is not a ChuchuClassifier instance")
        return model

    def _scores_by_class(self, X: np.ndarray) -> Dict[Any, np.ndarray]:
        X = np.asarray(X, float)
        return {cls: rs.model.score(X) for cls, rs in self.regions_.items()}

    def membership(self, X: np.ndarray) -> Dict[Any, np.ndarray]:
        """Máscara por clase: True si x está dentro de la región conformal de la clase."""
        scores = self._scores_by_class(X)
        return {cls: (s >= self.regions_[cls].threshold) for cls, s in scores.items()}

    def predict_proba(self, X: np.ndarray) -> Dict[Any, np.ndarray]:
        """Posterior proporcional: prior_c * score_c(x), normalizado en clases."""
        scores = self._scores_by_class(X)
        labels = list(self.regions_.keys())
        S = np.stack([scores[c] * self.priors_[c] for c in labels], axis=1)
        S = _safe_div(S, S.sum(axis=1, keepdims=True))
        return {c: S[:, i] for i, c in enumerate(labels)}

    def predict(self, X: np.ndarray, reject_if_outside: bool = True) -> np.ndarray:
        """Decisión dura: argmax posterior entre clases que aceptan el punto (si reject_if_outside)."""
        X = np.asarray(X, float)
        proba = self.predict_proba(X)
        mem = self.membership(X)
        labels = list(self.regions_.keys())
        P = np.stack([proba[c] for c in labels], axis=1)
        M = np.stack([mem[c] for c in labels], axis=1)
        if reject_if_outside:
            P = np.where(M, P, -np.inf)
        yhat_idx = np.argmax(P, axis=1)
        rej = ~np.isfinite(P).any(axis=1)
        yhat = np.array([labels[i] for i in yhat_idx], dtype=object)
        yhat[rej] = -1
        return yhat

    def predict_regions(self, X: ArrayLike) -> pd.DataFrame:
        """Return predicted labels and region ids for ``X``.

        ``region_id`` is the index of the class region (starting at 0) when the
        sample lies inside that class's region, and ``-1`` otherwise.  The
        returned dataframe is compatible with :meth:`membership`.
        """
        if isinstance(X, pd.DataFrame):
            index = X.index
            X_arr = X.to_numpy(dtype=float)
        else:
            index = None
            X_arr = np.asarray(X, float)

        labels = self.predict(X_arr)
        mem = self.membership(X_arr)
        label_list = list(self.regions_.keys())
        region_ids = np.full(labels.shape[0], -1, dtype=int)
        for idx, lab in enumerate(label_list):
            mask = (labels == lab) & mem[lab]
            region_ids[mask] = idx
        return pd.DataFrame({"label": labels.astype(int), "region_id": region_ids}, index=index)

    # ------------------------------------------------------------------
    def plot_classes(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ):
        """Plot class-wise regions overlaid with labelled data.

        Parameters
        ----------
        X, y:
            Data and corresponding class labels.
        feature_names:
            Optional names for each feature; defaults to ``x0``, ``x1``, ...
        """
        import matplotlib.pyplot as plt  # local import
        from itertools import combinations

        X_arr = np.asarray(X, float)
        y_arr = np.asarray(y)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        pairs = list(combinations(range(X_arr.shape[1]), 2))
        if not pairs:
            raise ValueError("plot_classes requires at least two features")
        fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))
        if len(pairs) == 1:
            axes = [axes]
        uniq = np.unique(y_arr)
        for ax, (i, j) in zip(axes, pairs):
            for cls in uniq:
                m = y_arr == cls
                ax.scatter(
                    X_arr[m, i],
                    X_arr[m, j],
                    s=10,
                    alpha=0.5,
                    label=str(cls),
                )
            for cls, reg in self.regions_.items():
                polys = hdr_polygons_2d(X_arr[:, [i, j]], reg, (0, 1))
                for poly in polys:
                    ax.plot(poly[:, 0], poly[:, 1], label=f"{cls}")
            if feature_names is not None and len(feature_names) > max(i, j):
                ax.set_xlabel(feature_names[i])
                ax.set_ylabel(feature_names[j])
            ax.legend()
        return fig, axes

    # ------------------------------------------------------------------
    def plot_pairs(
        self,
        X: ArrayLike,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ):
        """Plot 2D feature pairs with class regions.

        This basic visualisation draws data points and the HDR/MVS region for
        each class on every pair of features. Only 2-D trained models are
        supported.
        """
        import matplotlib.pyplot as plt  # local import to keep optional dependency
        from itertools import combinations

        X_arr = np.asarray(X, float)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        pairs = list(combinations(range(X_arr.shape[1]), 2))
        if not pairs:
            raise ValueError("plot_pairs requires at least two features")
        fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))
        if len(pairs) == 1:
            axes = [axes]
        for ax, (i, j) in zip(axes, pairs):
            ax.scatter(X_arr[:, i], X_arr[:, j], s=10, alpha=0.5, label="data")
            for cls, reg in self.regions_.items():
                polys = hdr_polygons_2d(X_arr[:, [i, j]], reg, (0, 1))
                for poly in polys:
                    ax.plot(poly[:, 0], poly[:, 1], label=f"{cls}")
            if feature_names is not None and len(feature_names) > max(i, j):
                ax.set_xlabel(feature_names[i])
                ax.set_ylabel(feature_names[j])
            ax.legend()
        return fig, axes

    def plot_pair_3d(self, *args, **kwargs):
        """3D pair plotting is not available for CHUCHU."""
        raise NotImplementedError("plot_pair_3d is not implemented for ChuchuClassifier")

    # ------------------ Optimización por Macro-F1 ------------------
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        density_grid: List[DensitySpec],
        coverage_grid: Iterable[float] = (0.80, 0.85, 0.90, 0.95),
        method: str = "hdr",
        n_splits: int = 5,
        random_state: int = 0,
        reject_if_outside: bool = True,
    ) -> Tuple[ChuchuConfig, float]:
        """
        Grid search para maximizar Macro-F1 (ignorando rechazos). Devuelve mejor config y su score.
        """
        X = np.asarray(X, float)
        y = np.asarray(y)
        best_cfg: Optional[ChuchuConfig] = None
        best_score = -np.inf
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for cov in coverage_grid:
            for dens in density_grid:
                f1s = []
                for tr, te in skf.split(X, y):
                    clf = ChuchuClassifier(
                        ChuchuConfig(method=method, coverage=cov, density=dens, random_state=random_state)
                    ).fit(X[tr], y[tr])
                    yhat = clf.predict(X[te], reject_if_outside=reject_if_outside)
                    f1s.append(macro_f1_ignore_rejects(y[te], yhat))
                mean_f1 = float(np.mean(f1s))
                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_cfg = ChuchuConfig(method=method, coverage=cov, density=dens, random_state=random_state)

        assert best_cfg is not None
        self.config = best_cfg
        self.fit(X, y)
        return best_cfg, best_score


class ChuchuRegressor:
    """
    Define una región HDR/MVS sobre X (no por clase) y predice con Nadaraya–Watson
    (o cualquier predictor que enchufes si lo deseas). Optimiza parámetros para
    minimizar MAPE con restricción de cobertura mínima.
    """

    __artifact_version__ = _ARTIFACT_VERSION

    def __init__(self, config: ChuchuConfig):
        self.config = config
        self.region_: Optional[RegionSpec] = None
        self.scaler_: Optional[StandardScaler] = None
        self.Xtr_: Optional[np.ndarray] = None
        self.ytr_: Optional[np.ndarray] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {"config": self.config}
        if deep and hasattr(self.config, "get_params"):
            params.update(self.config.get_params(deep=deep))
        return params

    def set_params(self, **params) -> "ChuchuRegressor":
        cfg_params = {k: v for k, v in params.items() if k != "config"}
        if "config" in params:
            self.config = params["config"]
        if cfg_params:
            if hasattr(self.config, "set_params"):
                self.config.set_params(**cfg_params)
            else:
                for name, value in cfg_params.items():
                    setattr(self.config, name, value)
        return self

    def _nw_predict(self, Xq: np.ndarray, density: BaseDensityModel) -> np.ndarray:
        """
        Nadaraya–Watson con kernel gaussiano en el espacio whitened del modelo.
        """
        mu = getattr(density, "mu_", None)
        W = getattr(density, "W_", None)
        if mu is None or W is None:
            mu = np.zeros(self.Xtr_.shape[1])
            W = np.eye(self.Xtr_.shape[1])

        Ztr = _apply_whiten(self.Xtr_, mu, W)
        Zte = _apply_whiten(np.asarray(Xq, float), mu, W)

        if isinstance(density, KDEDensityModel):
            n, d = Ztr.shape
            h = max(_scott(n, d) * density.bandwidth_factor, 1e-3)
        else:
            var = np.var(Ztr, axis=0).mean()
            h = max(np.sqrt(var), 1e-3)

        yhat = np.empty(len(Zte), float)
        for i, z in enumerate(Zte):
            dif = Ztr - z
            w = np.exp(-0.5 * np.sum(dif * dif, axis=1) / (h ** 2))
            yhat[i] = (w @ self.ytr_) / (w.sum() + _EPS)
        return yhat

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChuchuRegressor":
        cfg = self.config
        rng = np.random.RandomState(cfg.random_state)
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        tr, ca, _ = _train_calib_split(X, None, cfg.calib_frac, rng=rng)
        Xtr, Xca = X[tr], X[ca]

        model = _build_density_model(cfg.density or {"name": "kde"})
        lo, hi = _bounding_box(X)

        if cfg.method == "hdr":
            t, m = _hdr_region(Xtr, Xca, cfg.coverage, model)
        elif cfg.method == "mvs":
            t, m = _mvs_region(Xtr, Xca, cfg.coverage, model, lo=lo, hi=hi, rng=rng)
        else:
            raise ValueError("method debe ser 'hdr' o 'mvs'")

        self.region_ = RegionSpec(
            method=cfg.method,
            threshold=float(t),
            coverage_target=float(cfg.coverage),
            model=model,
            meta={"density": cfg.density or {"name": "kde"}},
        )
        self.Xtr_, self.ytr_ = Xtr, y[tr]
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_kwargs) -> np.ndarray:
        """Fit the regressor and return predictions on ``X``."""
        self.fit(X, y, **fit_kwargs)
        return self.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw scores from the region's density model."""
        return self.region_.model.score(np.asarray(X, float))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Alias for :meth:`decision_function`."""
        return self.decision_function(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_kwargs) -> np.ndarray:
        """Fit the regressor and return transformed scores on ``X``."""
        self.fit(X, y, **fit_kwargs)
        return self.transform(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist the regressor to ``filepath`` using joblib."""
        payload = {"__artifact_version__": self.__artifact_version__, "model": self}
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ChuchuRegressor":
        """Load a regressor saved with :meth:`save`."""
        payload = joblib.load(filepath)
        ver = payload.get("__artifact_version__")
        if ver != cls.__artifact_version__:
            raise ValueError(
                f"Artifact version mismatch: expected {cls.__artifact_version__}, got {ver}"
            )
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("Loaded object is not a ChuchuRegressor instance")
        return model

    def region_mask(self, X: np.ndarray) -> np.ndarray:
        s = self.region_.model.score(np.asarray(X, float))
        return s >= self.region_.threshold

    def predict(self, X: np.ndarray, reject_if_outside: bool = False) -> np.ndarray:
        yhat = self._nw_predict(np.asarray(X, float), self.region_.model)
        if reject_if_outside:
            m = self.region_mask(X)
            yhat = np.where(m, yhat, np.nan)
        return yhat

    def predict_regions(self, X: ArrayLike) -> pd.DataFrame:
        """Return predictions and region ids for ``X``.

        ``region_id`` is ``0`` when the sample lies inside the conformal region
        and ``-1`` otherwise.  The ``label`` column contains the regression
        prediction and is compatible with :meth:`region_mask`.
        """
        if isinstance(X, pd.DataFrame):
            index = X.index
            X_arr = X.to_numpy(dtype=float)
        else:
            index = None
            X_arr = np.asarray(X, float)
        preds = self.predict(X_arr, reject_if_outside=False)
        mask = self.region_mask(X_arr)
        region_ids = np.where(mask, 0, -1)
        return pd.DataFrame({"label": preds, "region_id": region_ids}, index=index)

    # ------------------------------------------------------------------
    def plot_classes(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ):
        """Plot regression region with points coloured by target value."""
        import matplotlib.pyplot as plt  # local import
        from itertools import combinations

        X_arr = np.asarray(X, float)
        y_arr = np.asarray(y, float)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        pairs = list(combinations(range(X_arr.shape[1]), 2))
        if not pairs:
            raise ValueError("plot_classes requires at least two features")
        fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))
        if len(pairs) == 1:
            axes = [axes]
        for ax, (i, j) in zip(axes, pairs):
            sc = ax.scatter(
                X_arr[:, i], X_arr[:, j], c=y_arr, s=10, alpha=0.5
            )
            polys = hdr_polygons_2d(X_arr[:, [i, j]], self.region_, (0, 1))
            for poly in polys:
                ax.plot(poly[:, 0], poly[:, 1], color="red", label="region")
            if feature_names is not None and len(feature_names) > max(i, j):
                ax.set_xlabel(feature_names[i])
                ax.set_ylabel(feature_names[j])
            fig.colorbar(sc, ax=ax)
            ax.legend()
        return fig, axes

    # ------------------------------------------------------------------
    def plot_pairs(
        self,
        X: ArrayLike,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ):
        """Plot 2D feature pairs with the regression region."""
        import matplotlib.pyplot as plt  # local import
        from itertools import combinations

        X_arr = np.asarray(X, float)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        pairs = list(combinations(range(X_arr.shape[1]), 2))
        if not pairs:
            raise ValueError("plot_pairs requires at least two features")
        fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))
        if len(pairs) == 1:
            axes = [axes]
        for ax, (i, j) in zip(axes, pairs):
            ax.scatter(X_arr[:, i], X_arr[:, j], s=10, alpha=0.5, label="data")
            polys = hdr_polygons_2d(X_arr[:, [i, j]], self.region_, (0, 1))
            for poly in polys:
                ax.plot(poly[:, 0], poly[:, 1], color="red", label="region")
            if feature_names is not None and len(feature_names) > max(i, j):
                ax.set_xlabel(feature_names[i])
                ax.set_ylabel(feature_names[j])
            ax.legend()
        return fig, axes

    def plot_pair_3d(self, *args, **kwargs):
        """3D pair plotting is not implemented for CHUCHU."""
        raise NotImplementedError("plot_pair_3d is not implemented for ChuchuRegressor")

    # ------------------ Optimización por MAPE ------------------
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        density_grid: List[DensitySpec],
        coverage_grid: Iterable[float] = (0.75, 0.80, 0.85, 0.90),
        method: str = "hdr",
        n_splits: int = 5,
        random_state: int = 0,
        min_coverage: float = 0.60,
    ) -> Tuple[ChuchuConfig, float, float]:
        """
        Grid search para MINIMIZAR MAPE medio dentro de la región (con cobertura mínima).
        Devuelve mejor config + (MAPE, cobertura).
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        best_cfg: Optional[ChuchuConfig] = None
        best_mape, best_cov = np.inf, 0.0

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for cov in coverage_grid:
            for dens in density_grid:
                mape_list, cov_list = [], []
                for tr, te in kf.split(X):
                    reg = ChuchuRegressor(
                        ChuchuConfig(method=method, coverage=cov, density=dens, random_state=random_state)
                    ).fit(X[tr], y[tr])
                    inside = reg.region_mask(X[te])
                    coverage = float(np.mean(inside.astype(float)))
                    if coverage < min_coverage:
                        mape_list.append(np.inf); cov_list.append(coverage); continue
                    yhat = reg.predict(X[te], reject_if_outside=False)[inside]
                    this_mape = mape(y[te][inside], yhat)
                    mape_list.append(this_mape); cov_list.append(coverage)

                mean_mape = float(np.mean(mape_list))
                mean_cov = float(np.mean(cov_list))
                if (mean_mape < best_mape) or (np.isclose(mean_mape, best_mape) and mean_cov > best_cov):
                    best_mape, best_cov = mean_mape, mean_cov
                    best_cfg = ChuchuConfig(method=method, coverage=cov, density=dens, random_state=random_state)

        assert best_cfg is not None
        self.config = best_cfg
        self.fit(X, y)
        return best_cfg, best_mape, best_cov


# ---------------------------------------------------------------------
# Helpers de visualización 2D (pares de variables)
# ---------------------------------------------------------------------
def hdr_polygons_2d(
    X: np.ndarray,
    region: RegionSpec,
    pair: Tuple[int, int],
    grid_res: int = 600,
    pad: float = 0.25,
) -> List[np.ndarray]:
    """
    Proyecta a 2D (par de columnas) y devuelve paths del iso-nivel.
    Si la región proviene de 'score', evaluamos directamente el score del modelo en la rejilla.
    """
    import matplotlib.pyplot as plt  # local import para no forzar dependencia headless
    from matplotlib.path import Path

    X = np.asarray(X, float)
    Xp = X[:, list(pair)]
    x, y = Xp[:, 0], Xp[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xr, yr = xmax - xmin, ymax - ymin
    xmin -= pad * xr; xmax += pad * xr
    ymin -= pad * yr; ymax += pad * yr
    xx = np.linspace(xmin, xmax, grid_res)
    yy = np.linspace(ymin, ymax, grid_res)
    Xg, Yg = np.meshgrid(xx, yy)
    G = np.c_[Xg.ravel(), Yg.ravel()]

    # Evaluación del score en la rejilla
    Z = region.model.score(G).reshape(Xg.shape)

    # Extrae segmentos al nivel 'threshold'
    fig, ax = plt.subplots()
    cs = ax.contour(Xg, Yg, Z, levels=[region.threshold])
    segs = cs.allsegs[0] if cs.allsegs else []
    plt.close(fig)

    polys: List[np.ndarray] = []
    for v in segs:
        v = np.asarray(v)
        if v.shape[0] >= 3:
            if np.linalg.norm(v[0] - v[-1]) > 1e-10:
                v = np.vstack([v, v[0]])
            polys.append(v)
    return polys
