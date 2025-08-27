# region_interpretability.py
# -------------------------------------------
# Herramientas para convertir ClusterRegion(s)
# en tarjetas interpretables y reglas legibles.
# -------------------------------------------

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Optional
import logging
from contextlib import contextmanager
import numpy as np

try:
    import pandas as _pd  # opcional
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


@contextmanager
def _np_no_warnings():
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        yield


def _safe_cov_rank(B):
    """
    Devuelve el rango de la matriz de covarianza de B (muestras x features)
    evitando warnings y manejando casos con pocas muestras.
    """
    if B is None or np.ndim(B) != 2:
        return 0
    m, d = B.shape
    if m < 2 or d < 2:
        return 0
    with _np_no_warnings():
        cov = np.cov(B, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return 0
        return int(np.linalg.matrix_rank(cov))


def _fallback_dims_from_dirs(directions, n_features, k=2):
    """
    Si la proyección PCA/2D no es posible, elige k ejes por peso en 'directions'.
    """
    if directions is None or directions.size == 0:
        return tuple(range(min(k, n_features)))
    w = np.max(np.abs(directions), axis=0)
    idx = np.argsort(w)[-k:]
    return tuple(idx.tolist())


# ==================== Utilidades geométricas ====================

def _rdp(points: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    """Ramer–Douglas–Peucker 2D para simplificar polilíneas."""
    def perp_dist(pt, a, b):
        ab = b - a
        den = np.linalg.norm(ab)
        if den < 1e-15:
            return np.linalg.norm(pt - a)
        return abs(np.cross(ab, a - pt)) / den

    def rec(pts):
        if pts.shape[0] <= 2:
            return pts
        a, b = pts[0], pts[-1]
        dmax, idx = 0.0, 0
        for i in range(1, pts.shape[0] - 1):
            d = perp_dist(pts[i], a, b)
            if d > dmax:
                dmax, idx = d, i
        if dmax > epsilon:
            left = rec(pts[:idx+1]); right = rec(pts[idx:])
            return np.vstack([left[:-1], right])
        else:
            return np.vstack([a, b])

    return rec(points)


def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Monotone chain; devuelve el hull CCW sin repetir el primer punto."""
    pts = np.unique(pts, axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _halfspaces_from_polygon(poly: np.ndarray, inside: np.ndarray) -> List[Tuple[float,float,float]]:
    """Devuelve (a,b,c) tales que a*x + b*y <= c define el interior de 'poly'."""
    hs = []
    m = poly.shape[0]
    for i in range(m):
        p0, p1 = poly[i], poly[(i+1) % m]
        v = p1 - p0
        n = np.array([-v[1], v[0]])      # normal hacia la izquierda
        a, b = n
        c = a*p0[0] + b*p0[1]
        if a*inside[0] + b*inside[1] > c:  # invierte si el centro viola
            a, b, c = -a, -b, -c
        hs.append((a, b, c))
    return hs


def _rule_text(a: float, b: float, c: float, xname: str, yname: str, decimals: int = 2) -> str:
    eps = 1e-12
    if abs(b) <= 1e-9 and abs(a) <= 1e-9:
        return "0 ≤ 0"
    if abs(b) <= 1e-9:  # recta vertical: a*x <= c
        sense = "<=" if a > 0 else ">="
        return f"{xname} {sense} {round(c/(a if a!=0 else eps), decimals)}"
    if abs(a) <= 1e-9:  # recta horizontal: b*y <= c
        sense = "<=" if b > 0 else ">="
        return f"{yname} {sense} {round(c/(b if b!=0 else eps), decimals)}"
    if abs(b) >= 1.2 * abs(a):
        m = -a / b
        t = c / b
        sense = "<=" if b > 0 else ">="
        return f"{yname} {sense} {round(m,decimals)}·{xname} + {round(t,decimals)}"
    elif abs(a) > 1.2 * abs(b):
        m = -b / a
        t = c / a
        sense = "<=" if a > 0 else ">="
        return f"{xname} {sense} {round(m,decimals)}·{yname} + {round(t,decimals)}"
    else:
        m = -a / b
        t = c / b
        sense = "<=" if b > 0 else ">="
        return f"{yname} {sense} {round(m,decimals)}·{xname} + {round(t,decimals)}"


# ==================== Núcleo de interpretabilidad ====================

def _extract_region(reg: Any) -> Tuple[int, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Admite dict u objeto (p.ej., ClusterRegion).

    Devuelve ``cluster_id``, ``label``, ``center``, ``directions`` y ``radii``.
    ``cluster_id`` cae a ``label`` si el objeto no lo expone explícitamente.
    """
    if isinstance(reg, dict):
        cid = int(reg.get("cluster_id", reg.get("label", 0)))
        label = reg.get("label", cid)
        try:
            label = int(label)
        except Exception:
            pass
        center = np.asarray(reg["center"], float)
        directions = np.asarray(reg["directions"], float)
        radii = np.asarray(reg["radii"], float)
    else:
        cid = int(getattr(reg, "cluster_id", getattr(reg, "label")))
        label = getattr(reg, "label", cid)
        try:
            label = int(label)
        except Exception:
            pass
        center = np.asarray(getattr(reg, "center"), float)
        directions = np.asarray(getattr(reg, "directions"), float)
        radii = np.asarray(getattr(reg, "radii"), float)
    return cid, label, center, directions, radii


def _sanitize_directions(
    dirs: np.ndarray, radii: np.ndarray, eps_r: float = 1e-8, cos_tol: float = 0.999
) -> Tuple[np.ndarray, np.ndarray]:
    """
    - Quita radios ~0.
    - Colapsa direcciones casi idénticas en el MISMO sentido (cos > cos_tol).
    - Mantiene direcciones opuestas (necesarias para ambos extremos).
    """
    mask = radii > eps_r
    dirs, radii = dirs[mask], radii[mask]
    if dirs.size == 0:
        return dirs, radii
    U = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-15)
    keep = []
    for i, v in enumerate(U):
        duplicate = False
        for j in keep:
            if float(np.dot(v, U[j])) > cos_tol:  # mismo sentido
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return dirs[keep], radii[keep]


def _boundary_points(center: np.ndarray, dirs: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Devuelve puntos del contorno en D dimensiones."""
    n = min(len(dirs), len(radii))
    if n == 0:
        return np.empty((0, center.size))
    return center[None, :] + dirs[:n] * radii[:n, None]


def _axis_box(points: np.ndarray, q: float = 0.05) -> List[Tuple[float, float]]:
    """Caja robusta por dimensión usando cuantiles [q, 1-q]."""
    lo = np.quantile(points, q, axis=0)
    hi = np.quantile(points, 1 - q, axis=0)
    return list(zip(lo.tolist(), hi.tolist()))


def _pick_best_pairs(points: np.ndarray, k: int = 2) -> List[Tuple[int, int, float]]:
    """
    Selecciona k proyecciones 2D con mayor poder explicativo:
    rango_i * rango_j penalizado por "linealidad" (razón de eigenvalores).
    """
    if points.shape[0] < 2:
        return []
    D = points.shape[1]
    rng = points.max(0) - points.min(0)
    scores: List[Tuple[int,int,float]] = []
    for i in range(D):
        for j in range(i+1, D):
            area = float(rng[i] * rng[j])
            if area < 1e-12:
                continue
            P = points[:, [i, j]]
            with _np_no_warnings():
                cov = np.cov(P.T)
            w, _ = np.linalg.eigh(cov)
            if (w.sum() <= 0):
                continue
            lineyness = float(w.max() / (w.sum()))  # 1.0 => nube casi línea
            score = area * (1.0 - 0.5 * lineyness)
            scores.append((i, j, score))
    scores.sort(key=lambda t: t[2], reverse=True)
    return scores[:k]


def _projection_rules(
    points_2d: np.ndarray, center_2d: np.ndarray, decimals: int = 2, max_rules: int = 5
) -> List[Tuple[float, float, float]]:
    """Hull -> RDP -> halfspaces. Devuelve a*x + b*y <= c."""
    if points_2d.shape[0] < 3:
        return []
    hull = _convex_hull_2d(points_2d)
    if hull.shape[0] >= 4:
        hull = _rdp(hull, epsilon=0.03 * np.mean(np.ptp(hull, axis=0)))
    if hull.shape[0] < 3:
        return []
    return _halfspaces_from_polygon(hull, center_2d)[:max_rules]


def _fmt_rules(hs: List[Tuple[float,float,float]], xname: str, yname: str, decimals: int = 2) -> List[str]:
    return [_rule_text(a, b, c, xname, yname, decimals) for (a, b, c) in hs]


def _ensure_names(D: int, feature_names: Optional[List[str]]) -> List[str]:
    """Ajusta nombres a D: genera, rellena o trunca según convenga."""
    if feature_names is None:
        return [f"feat {i}" for i in range(D)]
    names = list(feature_names)
    if len(names) < D:
        names += [f"feat {i}" for i in range(len(names), D)]
    elif len(names) > D:
        names = names[:D]
    return names


# ==================== API principal ====================

class RegionInterpreter:
    """
    Convierte ClusterRegion(s) en tarjetas interpretables:
    - cluster_id y label
    - headline humano
    - reglas por eje (caja robusta)
    - reglas por proyección 2D (3–5 desigualdades lineales)
    - notas de edge cases (radios capados, dims casi fijas, degeneración)
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        q_box: float = 0.05,
        k_pairs: int = 2,
        decimals: int = 2,
        cap_threshold: float = 6.393,
        near_const_tol: float = 0.12,
        *,
        inverse_transform=None,
        feature_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        include_center_in_box: bool = True,
    ) -> None:
        self.feature_names = feature_names
        self.q_box = q_box
        self.k_pairs = k_pairs
        self.decimals = decimals
        self.cap_threshold = cap_threshold
        self.near_const_tol = near_const_tol
        self.inverse_transform = inverse_transform
        self.feature_bounds = (
            None if feature_bounds is None else np.asarray(feature_bounds, float)
        )
        self.include_center_in_box = include_center_in_box

    def summarize(self, regions: Sequence[Any] | Any) -> List[Dict[str, Any]]:
        """Acepta lista de objetos/dicts o un solo objeto."""
        if not isinstance(regions, (list, tuple)):
            regions = [regions]

        out: List[Dict[str, Any]] = []
        for reg in regions:
            cid, label, center, dirs, radii = _extract_region(reg)

            # limpieza de direcciones / radios
            dirs, radii = _sanitize_directions(dirs, radii)
            if dirs.size == 0:
                center_out = (
                    self.inverse_transform(center[None, :]).ravel()
                    if self.inverse_transform is not None
                    else center
                )
                out.append(dict(
                    cluster_id=int(cid),
                    label=label,
                    center=[round(float(x), self.decimals) for x in center_out.tolist()],
                    headline="Región degenerada (sin radios útiles).",
                    box_rules=[],
                    pairwise_rules=[],
                    notes=["Todos los radios ~ 0."]
                ))
                continue

            # puntos de contorno y metadatos
            B = _boundary_points(center, dirs, radii)    # (N, D)
            D = B.shape[1]
            names = _ensure_names(D, self.feature_names)

            # llevar a espacio original si se provee inverse_transform
            if self.inverse_transform is not None:
                B_orig = self.inverse_transform(B)
                center_orig = self.inverse_transform(center[None, :]).ravel()
            else:
                B_orig = B
                center_orig = center

            # 1) caja por ejes (robusta)
            box_pts = B_orig
            if self.include_center_in_box:
                box_pts = np.vstack([box_pts, center_orig[None, :]])

            box = _axis_box(box_pts, q=self.q_box)
            box_rules, near_const = [], []
            for i, (lo, hi) in enumerate(box):
                if self.feature_bounds is not None:
                    lo = max(lo, float(self.feature_bounds[i, 0]))
                    hi = min(hi, float(self.feature_bounds[i, 1]))
                width = hi - lo
                if width < self.near_const_tol:
                    near_const.append((i, (lo + hi) / 2))
                box_rules.append(f"{names[i]} ∈ [{round(lo, self.decimals)}, {round(hi, self.decimals)}]")

            # 2) proyecciones más informativas
            pair_rules = []
            if D >= 2:
                pairs = _pick_best_pairs(B_orig, k=self.k_pairs)
                for (i, j, _) in pairs:
                    P = B_orig[:, [i, j]]
                    hs = _projection_rules(P, center_orig[[i, j]], decimals=self.decimals, max_rules=5)
                    pair_rules.append({
                        "pair": (names[i], names[j]),
                        "rules": _fmt_rules(hs, names[i], names[j], self.decimals)
                    })

            # 3) headline humano (bajo/medio/alto + dimensiones casi fijas)
            def pos_word(i):
                lo, hi = box[i]; val = center_orig[i]
                p = (val - lo) / (hi - lo + 1e-12)
                return "bajo" if p <= 0.25 else ("alto" if p >= 0.75 else "medio")

            words = [f"{names[i]} {pos_word(i)}" for i in range(min(D, 3))]
            if near_const:
                words.append(", ".join([f"{names[i]} ≈ {round(v, self.decimals)}" for i, v in near_const[:2]]))
            headline = " / ".join(words)

            # 4) notas (edge cases)
            notes: List[str] = []
            if float(np.mean(radii >= self.cap_threshold)) > 0.25:
                notes.append("Muchos radios en el **límite** del grid → frontera posiblemente abierta.")
            if near_const:
                notes.append("Alguna dimensión queda **casi fija** (ancho muy pequeño).")
            if D >= 2 and _safe_cov_rank(B_orig) < 2:
                notes.append("Alguna proyección es degenerada (varianza efectiva baja).")

            center_out = [round(float(x), self.decimals) for x in center_orig.tolist()]

            out.append(dict(
                cluster_id=int(cid),
                label=label,
                center=center_out,
                headline=headline,
                box_rules=box_rules,
                pairwise_rules=pair_rules,
                notes=notes
            ))
        return out

    # ---------- utilidades de salida ----------

    @staticmethod
    def to_dataframe(cards: List[Dict[str, Any]]):
        """Convierte la lista de tarjetas a DataFrame (si hay pandas)."""
        if not _HAS_PANDAS:
            raise ImportError("Pandas no está instalado. Instálalo o usa la salida en dicts.")
        rows = []
        for c in cards:
            base = {
                "cluster_id": c["cluster_id"],
                "label": c["label"],
                "center": c["center"],
                "headline": c["headline"],
                "notes": "; ".join(c.get("notes", []))
            }
            # expandir reglas por ejes
            for i, r in enumerate(c.get("box_rules", [])):
                base[f"box_rule_{i+1}"] = r
            # expandir reglas por pares
            for k, pr in enumerate(c.get("pairwise_rules", [])):
                base[f"pair_{k+1}"] = " / ".join(pr["pair"])
                for j, rr in enumerate(pr.get("rules", [])):
                    base[f"pair_{k+1}_rule_{j+1}"] = rr
            rows.append(base)
        return _pd.DataFrame(rows)

    @staticmethod
    def pretty_print(cards: List[Dict[str, Any]]) -> None:
        """Impresión amigable en consola."""
        for c in cards:
            logger.info("\n=== Región %s (label %s) ===", c['cluster_id'], c['label'])
            logger.info("Center: %s", c["center"])
            logger.info("Headline: %s", c["headline"])
            if c.get("box_rules"):
                logger.info("Caja por ejes:")
                for r in c["box_rules"]:
                    logger.info(" • %s", r)
            if c.get("pairwise_rules"):
                logger.info("Proyecciones clave:")
                for pr in c["pairwise_rules"]:
                    logger.info("  %s:", pr["pair"])
                    for r in pr["rules"]:
                        logger.info("   - %s", r)
            if c.get("notes"):
                logger.info("Notas: %s", "; ".join(c["notes"]))
