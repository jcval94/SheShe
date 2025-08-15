import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, Dict, Optional, Literal
import time
import warnings

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

# --------- opcionales: LightGBM/SHAP/EBM ------------
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    _HAS_EBM = True
except Exception:
    _HAS_EBM = False
# -----------------------------------------------------


class SubspaceScout:
    """
    Descubre subespacios útiles (pares, tríos, …) para correr SheShe solo donde vale la pena.
    Ahora con control de crecimiento (beam search + muestreo aleatorio).
    """

    def __init__(self,
                 model_method: Optional[Literal['lightgbm', 'ebm']] = None,
                 max_order: int = 3,
                 n_bins: int = 8,
                 top_m: int = 20,
                 branch_per_parent: int = 5,          # legacy: nº máx. extensiones por padre (tras ordenar por score)
                 density_occup_min: float = 0.03,
                 min_support: int = 30,
                 sample_size: Optional[int] = 4096,
                 task: Literal['classification', 'regression'] = 'classification',
                 random_state: int = 0,
                 # ---- NUEVOS PARÁMETROS ANTI-EXPLOSIÓN ----
                 base_pairs_limit: int = 12,          # nº máx. de pares semilla para órdenes >=3
                 beam_width: int = 12,                # nº máx. de combos que pasan a la siguiente capa
                 extend_candidate_pool: Optional[int] = 16,  # nº de features candidatas aleatorias por padre
                 marginal_gain_min: float = 1e-3,     # ganancia mínima de sinergia para aceptar extensión
                 max_eval_per_order: Optional[int] = 1000,   # tope de evaluaciones MI por orden
                 time_budget_s: Optional[float] = None, # presupuesto total de tiempo (seg.) para fit()
                 objective: Literal['mi_joint','mi_synergy'] = 'mi_joint',
                 min_per_order: int = 1   # ← garantiza al menos este nº por orden
                 ):
        self.model_method = model_method
        self.max_order = max_order
        self.n_bins = n_bins
        self.top_m = top_m
        self.branch_per_parent = branch_per_parent
        self.density_occup_min = density_occup_min
        self.min_support = min_support
        self.sample_size = sample_size
        self.task = task
        self.random_state = random_state

        # Nuevos
        self.base_pairs_limit = base_pairs_limit
        self.beam_width = beam_width
        self.extend_candidate_pool = extend_candidate_pool
        self.marginal_gain_min = marginal_gain_min
        self.max_eval_per_order = max_eval_per_order
        self.time_budget_s = time_budget_s
        self.objective = objective
        self.min_per_order = min_per_order

        # Internos
        self._disc = None
        self._Xd = None
        self._X_used = None
        self._y_used = None
        self._mi_single = None
        self._bins_per_feature = None
        self._mi_joint_cache: Dict[Tuple[int, ...], float] = {}  # cache MI conjunta
        self.results_: List[Dict] = []

    # ---------------------- Utils MI ----------------------

    # def _discretize(self, X):
    #     self._disc = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="quantile")
    #     Xd = self._disc.fit_transform(X).astype(int)
    #     return Xd

    def _mi_single_all(self, Xd, y):
        d = Xd.shape[1]
        mi = np.zeros(d)
        if self.task == 'classification':
            for i in range(d):
                mi[i] = mutual_info_classif(Xd[:, [i]], y, discrete_features=True,
                                            random_state=self.random_state)[0]
        else:
            for i in range(d):
                mi[i] = mutual_info_regression(Xd[:, [i]], y, discrete_features=True,
                                               random_state=self.random_state)[0]
        return mi

    # def _mi_joint(self, Xd_cols, y, key: Tuple[int, ...]):
    #     """MI(Y ; joint(X_S)) con cache por 'key'=tuple(sorted(S))."""
    #     if key in self._mi_joint_cache:
    #         return self._mi_joint_cache[key]
    #     base = self.n_bins
    #     joint = Xd_cols[:, 0].copy()
    #     for k in range(1, Xd_cols.shape[1]):
    #         joint = joint * base + Xd_cols[:, k]
    #     joint = joint.reshape(-1, 1)
    #     if self.task == 'classification':
    #         val = mutual_info_classif(joint, y, discrete_features=True,
    #                                   random_state=self.random_state)[0]
    #     else:
    #         val = mutual_info_regression(joint, y, discrete_features=True,
    #                                      random_state=self.random_state)[0]
    #     self._mi_joint_cache[key] = float(val)
    #     return float(val)

    def _synergy(self, S: Tuple[int, ...]):
        """Sinergia = MI(Y;X_S) - sum_i MI(Y;X_i). Generaliza a |S|>=2."""
        S = tuple(int(v) for v in S)  # normaliza np.int64 -> int
        XdS = self._Xd[:, S]
        mi_joint = self._mi_joint(XdS, self._y_used, key=S)
        mi_sum = self._mi_single[list(S)].sum()
        return mi_joint - mi_sum, mi_joint

    # def _occupancy_ok(self, S: Tuple[int, ...]):
    #     XdS = self._Xd[:, S]
    #     n, k = XdS.shape[0], len(S)
    #     dims = (self.n_bins,) * k
    #     joint = np.ravel_multi_index(XdS.T, dims=dims, mode='wrap')
    #     occ_ratio = np.unique(joint).size / np.prod(dims)

    #     # ocupación esperada bajo Poisson: 1 - exp(-n/total_bins)
    #     total_bins = float(np.prod(dims))
    #     expected = 1.0 - np.exp(-n / total_bins)

    #     # umbral mínimo: max( fracción base escalada por orden, fracción de lo esperado )
    #     base = self.density_occup_min * (2 ** (2 - k))  # relaja con el orden
    #     min_required = max(base, 0.5 * expected)

    #     return (occ_ratio >= min_required) and (n >= self.min_support)

    def _discretize(self, X):
        X = np.asarray(X, float).copy()
        n, d = X.shape

        # imputación simple por mediana (evita NaN en el binning)
        for j in range(d):
            col = X[:, j]
            m = np.isfinite(col)
            if not m.all():
                med = np.nanmedian(col[m]) if m.any() else 0.0
                col[~m] = med
                X[:, j] = col

        # nº de valores distintos por feature
        n_unique = np.array([np.unique(X[:, j]).size for j in range(d)], dtype=int)

        keep = np.where(n_unique >= 2)[0]         # discretizables
        drop = np.where(n_unique <  2)[0]         # constantes

        bins = np.ones(d, dtype=int)              # por defecto 1 categoría (constantes)
        bins[keep] = np.minimum(self.n_bins, np.maximum(2, n_unique[keep] - 1))

        # jitter muy pequeño si hay empates extremos para evitar edges duplicados
        rng = np.random.default_rng(self.random_state + 13)
        for j in keep:
            tie_ratio = 1.0 - (n_unique[j] / n)
            if tie_ratio > 0.98:  # muuuchos empates
                s = np.nanstd(X[:, j]) or 1.0
                X[:, j] = X[:, j] + rng.normal(0.0, 1e-9 * s, size=n)

        Xd = np.zeros((n, d), dtype=int)
        if keep.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                disc = KBinsDiscretizer(
                    n_bins=bins[keep],
                    encode="ordinal",
                    strategy="quantile"
                )
                Xd_keep = disc.fit_transform(X[:, keep]).astype(int)
            Xd[:, keep] = Xd_keep

        # guarda bins por feature
        self._bins_per_feature = bins
        return Xd

    # --- reemplaza _mi_joint: usa dims por-feature (evita 'base = self.n_bins') ---
    def _mi_joint(self, Xd_cols, y, key: Tuple[int, ...]):
        if key in self._mi_joint_cache:
            return self._mi_joint_cache[key]
        dims = tuple(int(self._bins_per_feature[k]) for k in key)  # p.ej. (8,7,1,...)
        # ojo: si alguna dim es 1, todos los índices deben ser 0 (nuestro Xd ya lo es)
        joint = np.ravel_multi_index(Xd_cols.T, dims=dims, mode='raise').reshape(-1, 1)
        if self.task == 'classification':
            val = mutual_info_classif(joint, y, discrete_features=True,
                                      random_state=self.random_state)[0]
        else:
            val = mutual_info_regression(joint, y, discrete_features=True,
                                        random_state=self.random_state)[0]
        self._mi_joint_cache[key] = float(val)
        return float(val)

    # --- reemplaza _occupancy_ok: también con dims por-feature ---
    def _occupancy_ok(self, S: Tuple[int, ...]):
        XdS = self._Xd[:, S]
        dims = tuple(int(self._bins_per_feature[k]) for k in S)
        n = XdS.shape[0]; k = len(S)
        joint = np.ravel_multi_index(XdS.T, dims=dims, mode='raise')
        occ_ratio = np.unique(joint).size / float(np.prod(dims))

        # ocupación esperada bajo Poisson
        total_bins = float(np.prod(dims))
        expected = 1.0 - np.exp(-n / total_bins)

        base = self.density_occup_min * (2 ** (2 - k))  # relaja con el orden
        min_required = max(base, 0.5 * expected)
        return (occ_ratio >= min_required) and (n >= self.min_support)

        
    # ------------------- LightGBM + SHAP (pares) -------------------
    def _score_pairs_shap(self, X, y) -> Dict[Tuple[int, int], float]:
        if not (_HAS_LGB and _HAS_SHAP):
            raise RuntimeError("LightGBM/SHAP no disponibles.")

        X = np.asarray(X)
        y = np.asarray(y)

        # --- 1) Saneado: quitar columnas constantes/NaN ---
        col_var = np.nanstd(X, axis=0)
        keep = np.where(col_var > 0)[0]
        if keep.size < 2 or np.unique(y).size < 2:
            return {}
        X2 = X[:, keep].astype(np.float32, copy=False)
        d2 = X2.shape[1]

        # Submuestreo opcional para SHAP
        if self.sample_size and X2.shape[0] > self.sample_size:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X2.shape[0], self.sample_size, replace=False)
            Xs = X2[idx]
            ys = y[idx]
        else:
            Xs = X2
            ys = y

        # --- 2) Modelo LightGBM robusto para evitar "no further splits" ---
        n = Xs.shape[0]
        # Evita bagging/colsample agresivo si n es chico
        subsample = 1.0 if n < 500 else 0.8
        colsample = 1.0 if d2 < 20 else 0.8
        min_data_in_leaf = max(10, int(0.01 * n))

        if self.task == 'classification':
            model = lgb.LGBMClassifier(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                subsample=subsample,
                colsample_bytree=colsample,
                min_data_in_leaf=min_data_in_leaf,
                min_sum_hessian_in_leaf=1e-3,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                subsample=subsample,
                colsample_bytree=colsample,
                min_data_in_leaf=min_data_in_leaf,
                min_sum_hessian_in_leaf=1e-3,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )

        model.fit(Xs, ys)

        expl = shap.TreeExplainer(model)
        inter = expl.shap_interaction_values(Xs)

        # --- 3) Reducir a matriz (d2, d2) de forma robusta ---
        if isinstance(inter, list):
            M = np.mean([np.abs(m).mean(axis=0) for m in inter], axis=0)  # (d2,d2)
        else:
            A = np.abs(np.asarray(inter))
            if A.ndim == 3:          # (n,d2,d2)
                M = A.mean(axis=0)
            elif A.ndim == 4:        # (n,d2,d2,C)
                M = A.mean(axis=(0, 3))
            else:
                axes = tuple(range(A.ndim - 2))
                M = A.mean(axis=axes)  # (d2,d2)

        # --- 4) Mapear de vuelta a índices originales y construir scores ---
        scores = {}
        for a in range(d2):
            ia = int(keep[a])
            for b in range(a + 1, d2):
                ib = int(keep[b])
                val = float(np.asarray(M[a, b]).mean())
                scores[(ia, ib)] = val

        return scores


    # ------------------------- EBM (pares) -------------------------

    def _score_pairs_ebm(self, X, y) -> Dict[Tuple[int, int], float]:
        if not _HAS_EBM:
            raise RuntimeError("EBM no disponible (interpret).")

        # Interacciones limitadas para velocidad; solo entre top_m features:
        n_feats = X.shape[1]
        max_pairs_among_topm = max(1, min(self.top_m * (self.top_m - 1) // 2, 32))
        n_interactions = min(20, max_pairs_among_topm)  # cap razonable

        if self.task == 'classification':
            ebm = ExplainableBoostingClassifier(
                random_state=self.random_state,
                interactions=n_interactions
            )
        else:
            ebm = ExplainableBoostingRegressor(
                random_state=self.random_state,
                interactions=n_interactions
            )

        ebm.fit(X, y)

        # --- Robustez de atributos entre versiones ---
        term_feats = getattr(ebm, "term_features_", None)
        # term_scores_: lista con arrays (para multiclase suele ser [C, ...])
        term_scores = getattr(ebm, "term_scores_", None)
        if term_scores is None:
            term_scores = getattr(ebm, "additive_terms_", None)

        if term_feats is None or term_scores is None:
            raise RuntimeError("EBM no expone term_features_/term_scores_ en esta versión.")

        scores: Dict[Tuple[int, int], float] = {}
        for k, feats in enumerate(term_feats):
            # feats: índices de features que componen el término k
            if len(feats) != 2:
                continue  # solo pares
            sc_arr = np.asarray(term_scores[k])

            # Importancia del término = magnitud media de su contribución
            # (promedia sobre clases si multiclase y sobre bins):
            imp = float(np.mean(np.abs(sc_arr)))

            i, j = sorted(int(f) for f in feats)
            scores[(i, j)] = imp

        return scores



    def _mi_joint_val(self, S: Tuple[int, ...]) -> float:
        S = tuple(int(v) for v in S)
        XdS = self._Xd[:, S]
        return self._mi_joint(XdS, self._y_used, key=S)

    def _objective_gain(self, parent, S_new):
        mi_parent = self._mi_joint_val(parent)
        mi_new = self._mi_joint_val(S_new)
        if self.objective == 'mi_joint':
            gain = mi_new - mi_parent
            syn_new = mi_new - self._mi_single[list(S_new)].sum()
        else:
            syn_parent = mi_parent - self._mi_single[list(parent)].sum()
            syn_new = mi_new - self._mi_single[list(S_new)].sum()
            gain = syn_new - syn_parent
        return float(gain), float(mi_new), float(syn_new)


    # --------------------------- FIT -------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X); y = np.asarray(y)
        # muestreo opcional para velocidad
        if self.sample_size and X.shape[0] > self.sample_size:
            if self.task == 'classification':
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=None,
                                                  train_size=self.sample_size,
                                                  random_state=self.random_state)
                idx = next(splitter.split(X, y))[0]
            else:
                splitter = ShuffleSplit(n_splits=1, test_size=None,
                                        train_size=self.sample_size,
                                        random_state=self.random_state)
                idx = next(splitter.split(X))[0]
            X_use, y_use = X[idx], y[idx]
        else:
            X_use, y_use = X, y

        self._X_used, self._y_used = X_use, y_use
        self._Xd = self._discretize(X_use)
        self._mi_single = self._mi_single_all(self._Xd, y_use)

        d = X.shape[1]
        # top_m por MI individual (rápido)
        top_feats = np.argsort(self._mi_single)[::-1][:min(self.top_m, d)].astype(int)

        results: Dict[Tuple[int, ...], float] = {}  # dedup: combo -> best score

        # ---------- pares ----------
        pair_scores: Dict[Tuple[int, int], float] = {}
        eval_count = 0

        if self.model_method is None:
            # MI sinergia para pares con ocupación
            for i, j in combinations(top_feats, 2):
                if self.time_budget_s and (time.perf_counter() - t0) > self.time_budget_s:
                    break
                S = (int(i), int(j))
                if not self._occupancy_ok(S):
                    continue
                syn, mij = self._synergy(S); eval_count += 1
                pair_scores[S] = float(syn)
                if self.max_eval_per_order and eval_count >= self.max_eval_per_order:
                    break

        elif self.model_method == 'lightgbm':
            pair_scores = self._score_pairs_shap(X_use, y_use)
            pair_scores = { (int(i),int(j)): s for (i,j), s in pair_scores.items()
                            if (i in top_feats and j in top_feats) and self._occupancy_ok((i,j)) }
        elif self.model_method == 'ebm':
            pair_scores = self._score_pairs_ebm(X_use, y_use)
            pair_scores = { (int(i),int(j)): s for (i,j), s in pair_scores.items()
                            if (i in top_feats and j in top_feats) and self._occupancy_ok((i,j)) }
        else:
            raise ValueError("model_method debe ser None | 'lightgbm' | 'ebm'.")

        pairs_sorted = sorted(pair_scores.items(), key=lambda t: t[1], reverse=True)
        # guarda pares y limita semillas para órdenes superiores
        for (i, j), sc in pairs_sorted:
            results[(i, j)] = max(sc, results.get((i, j), -np.inf))

        base_pairs = [fs for fs, _ in pairs_sorted[:max(1, self.base_pairs_limit)]]

        # ---------- tríos y superiores (beam + random subset) ----------

        if self.max_order >= 3 and base_pairs:
            rng = np.random.default_rng(self.random_state)
            cand_feats = list(top_feats)
            frontier = [tuple(sorted(p)) for p in base_pairs]

            for target_order in range(3, self.max_order + 1):
                if self.time_budget_s and (time.perf_counter() - t0) > self.time_budget_s:
                    break

                # ---------- EXPANSIÓN NORMAL ----------
                evals_this_order = 0
                candidates_global: List[Tuple[Tuple[int, ...], float, float]] = []  # (S_new, gain, mi_joint)
                seen_in_order = set()

                if len(frontier) > self.beam_width:
                    frontier = frontier[:self.beam_width]

                for parent in frontier:
                    if self.time_budget_s and (time.perf_counter() - t0) > self.time_budget_s:
                        break

                    feats_pool = [f for f in cand_feats if f not in parent]
                    pool = feats_pool
                    if self.extend_candidate_pool is not None and len(feats_pool) > self.extend_candidate_pool:
                        pool = rng.choice(feats_pool, size=self.extend_candidate_pool, replace=False)

                    scored_exts: List[Tuple[Tuple[int, ...], float, float]] = []
                    for k in pool:
                        if self.time_budget_s and (time.perf_counter() - t0) > self.time_budget_s:
                            break
                        S_new = tuple(sorted(parent + (int(k),)))
                        if S_new in seen_in_order:
                            continue
                        if not self._occupancy_ok(S_new):
                            continue
                        gain, mi_new, syn_new = self._objective_gain(parent, S_new)
                        evals_this_order += 1
                        if gain < max(self.marginal_gain_min, 0.0):
                            continue
                        scored_exts.append((S_new, gain, mi_new))
                        seen_in_order.add(S_new)
                        if self.max_eval_per_order and evals_this_order >= self.max_eval_per_order:
                            break

                    scored_exts.sort(key=lambda t: (t[1], t[2]), reverse=True)
                    candidates_global.extend(scored_exts[:self.branch_per_parent])
                    if self.max_eval_per_order and evals_this_order >= self.max_eval_per_order:
                        break

                candidates_global.sort(key=lambda t: (t[1], t[2]), reverse=True)
                candidates_global = candidates_global[:self.beam_width]

                # ---------- BACKOFF SI QUEDÓ VACÍO ----------
                def random_fill(n_wanted: int) -> List[Tuple[Tuple[int, ...], float, float]]:
                    # sample combos aleatorios de ese orden y rankear por MI conjunta
                    out = []
                    tries = 0
                    max_tries = n_wanted * 20
                    while len(out) < n_wanted and tries < max_tries:
                        S = tuple(sorted(rng.choice(cand_feats, size=target_order, replace=False).tolist()))
                        tries += 1
                        if S in seen_in_order or any(len(set(S).intersection(set(p))) == target_order for p in frontier):
                            # permite solapar, pero evita repetir exactamente
                            continue
                        if not self._occupancy_ok(S):
                            continue
                        miJ = self._mi_joint_val(S)
                        out.append((S, miJ, miJ))
                    out.sort(key=lambda t: t[1], reverse=True)
                    return out[:n_wanted]

                if len(candidates_global) < self.min_per_order:
                    # relaja: sin umbral y con pool ampliado
                    self.marginal_gain_min = 0.0
                    if self.extend_candidate_pool is not None:
                        self.extend_candidate_pool = max(self.extend_candidate_pool, self.beam_width * 2)

                    # segundo intento con reglas relajadas
                    if not candidates_global:
                        # toma al menos un relleno aleatorio razonable
                        candidates_global = random_fill(max(self.min_per_order, self.beam_width//2))

                # si aún seguimos cortos, fuerza mínimo por orden con random fill
                if len(candidates_global) < self.min_per_order:
                    extra = random_fill(self.min_per_order - len(candidates_global))
                    candidates_global.extend(extra)

                # ---------- actualizar resultados y frontera ----------
                new_frontier = []
                candidates_global.sort(key=lambda t: (t[1], t[2]), reverse=True)
                candidates_global = candidates_global[:max(self.beam_width, self.min_per_order)]

                for S_new, _, mi_new in candidates_global:
                    # puntúa por la métrica seleccionada; guarda lo mejor visto
                    if self.objective == 'mi_joint':
                        sc = mi_new
                        met = 'mi_joint'
                    else:
                        sc = self._synergy(S_new)[0]
                        met = 'mi_synergy'
                    results[S_new] = max(sc, results.get(S_new, -np.inf))
                    new_frontier.append(S_new)

                frontier = new_frontier
                if not frontier:
                    # si no quedó nada para seguir expandiendo, pero aún faltan órdenes,
                    # crea una frontera mínima aleatoria razonable
                    frontier = [S for S, _, _ in random_fill(max(1, self.min_per_order))]

        # ordenar y devolver en el formato original
        out = []
        for S, sc in results.items():
            out.append({'features': tuple(int(v) for v in S),
                        'order': len(S),
                        'score': float(sc),
                        'metric': self.model_method or 'mi_synergy'})
        self.results_ = sorted(out, key=lambda r: (r['order'], r['score']), reverse=True)
        return self.results_

    # ------------------- Helper para Fase B -------------------

    @staticmethod
    def fixed_vector_for_surface(X: np.ndarray, statistic: Literal['median', 'mean'] = 'median') -> np.ndarray:
        """Vector fijo para otras dims al dibujar/correr SheShe en un subespacio. Usa MEDIANA por defecto."""
        if statistic == 'median':
            return np.nanmedian(X, axis=0)
        else:
            return np.nanmean(X, axis=0)
