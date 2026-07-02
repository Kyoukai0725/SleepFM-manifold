"""Physical dynamics metrics on the UMAP-3D manifold trajectory."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from manifold_metrics import (
    METRIC_NAMES,
    SLIDING_STEP_EPOCHS,
    SLIDING_WINDOW_EPOCHS,
    _aggregate_std,
    _compute_four_metrics,
    _log_euclidean_project,
    embed_umap_3d,
    sliding_window_metrics,
)

PHYSICAL_METRIC_NAMES = ("MTV", "MTC", "MTE", "MSV", "MRR", "LZC")
DISTILL_METRIC_NAMES = METRIC_NAMES + ("MTV", "MTC", "MTE", "MSV", "MRR")
IRREGULARITY_METRIC_NAMES = ("MFI", "LZC")
ALL_METRIC_NAMES = METRIC_NAMES + PHYSICAL_METRIC_NAMES


def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Helper."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n < m + 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if r is None:
        r = 0.2 * sd
    if r <= 0 or not np.isfinite(r):
        return float("nan")

    def _count(m_len: int) -> float:
        n_pat = n - m_len + 1
        if n_pat < 1:
            return float("nan")
        patterns = np.lib.stride_tricks.sliding_window_view(x, m_len)
        total = 0.0
        for i in range(n_pat):
            dist = np.max(np.abs(patterns - patterns[i]), axis=1)
            total += float(np.sum(dist <= r) - 1)
        return total / (n_pat * (n_pat - 1))

    c_m = _count(m)
    c_m1 = _count(m + 1)
    if not np.isfinite(c_m) or not np.isfinite(c_m1) or c_m <= 0 or c_m1 <= 0:
        return float("nan")
    return float(-np.log(c_m1 / c_m))


def compute_mtv(X: np.ndarray) -> float:
    """Helper."""
    if len(X) < 2:
        return float("nan")
    vel = np.linalg.norm(np.diff(X, axis=0), axis=1)
    return float(np.mean(vel))


def compute_mtc(X: np.ndarray) -> float:
    """Helper."""
    if len(X) < 3:
        return float("nan")
    v = np.diff(X, axis=0)
    norms = np.linalg.norm(v, axis=1)
    angles: List[float] = []
    for i in range(len(v) - 1):
        if norms[i] < 1e-12 or norms[i + 1] < 1e-12:
            continue
        cos = float(np.dot(v[i], v[i + 1]) / (norms[i] * norms[i + 1]))
        angles.append(float(np.arccos(np.clip(cos, -1.0, 1.0))))
    return float(np.mean(angles)) if angles else float("nan")


def compute_mte(X: np.ndarray) -> float:
    """Helper."""
    if len(X) < 4:
        return float("nan")
    vel = np.linalg.norm(np.diff(X, axis=0), axis=1)
    return _sample_entropy(vel, m=2)


def _lempel_ziv_complexity(bits: np.ndarray) -> float:
    """Helper."""
    s = "".join("1" if int(b) else "0" for b in bits.ravel())
    n = len(s)
    if n < 8:
        return float("nan")

    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    while True:
        if s[i + k - 1] != s[l + k - 1]:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
        else:
            k += 1
            if l + k > n:
                c += 1
                break
    return float(c * np.log2(n) / n)


def compute_lzc(X: np.ndarray) -> float:
    """Helper."""
    if len(X) < 4:
        return float("nan")
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    if len(step) < 8:
        return float("nan")
    med = float(np.median(step))
    bits = (step > med).astype(int)
    return _lempel_ziv_complexity(bits)


def compute_msv(X: np.ndarray) -> float:
    """Helper."""
    if len(X) < 4:
        return float("nan")
    cov = np.cov(X.T)
    d = cov.shape[0]
    det = float(np.linalg.det(cov + 1e-8 * np.eye(d)))
    return float(np.log10(max(det, 1e-30)))


def compute_mrr(X: np.ndarray, eps_quantile: float = 0.10) -> float:
    """Helper."""
    n = len(X)
    if n < 6:
        return float("nan")
    diff = X[:, None, :] - X[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(D, np.inf)
    finite = D[np.isfinite(D)]
    if finite.size == 0:
        return float("nan")
    eps = float(np.quantile(finite, eps_quantile))
    if eps <= 0:
        eps = float(np.median(finite) * 0.05 + 1e-8)
    return float(np.mean(D < eps))


def _compute_physical_metrics(X_manifold: np.ndarray) -> Dict[str, float]:
    return {
        "MTV": compute_mtv(X_manifold),
        "MTC": compute_mtc(X_manifold),
        "MTE": compute_mte(X_manifold),
        "MSV": compute_msv(X_manifold),
        "MRR": compute_mrr(X_manifold),
        "LZC": compute_lzc(X_manifold),
    }


def sliding_window_all_metrics(
    X_umap: np.ndarray,
    window_epochs: int = SLIDING_WINDOW_EPOCHS,
    step_epochs: int = SLIDING_STEP_EPOCHS,
) -> List[Dict[str, float]]:
    """Helper."""
    X = np.asarray(X_umap, dtype=np.float64)
    n = len(X)
    if n < window_epochs:
        return []

    out: List[Dict[str, float]] = []
    for start in range(0, n - window_epochs + 1, step_epochs):
        chunk = X[start : start + window_epochs]
        X_m = _log_euclidean_project(chunk)
        m = {**_compute_four_metrics(X_m), **_compute_physical_metrics(X_m)}
        if all(np.isfinite(v) for v in m.values()):
            out.append(m)
    return out


def aggregate_all_metrics(window_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Helper."""
    empty: Dict[str, float] = {}
    for name in ALL_METRIC_NAMES:
        empty[f"{name}_std"] = float("nan")
        empty[f"{name}_mean"] = float("nan")
    empty["n_windows"] = float(len(window_metrics))
    if len(window_metrics) < 2:
        return empty

    for name in ALL_METRIC_NAMES:
        vals = np.array([w[name] for w in window_metrics], dtype=np.float64)
        empty[f"{name}_std"] = float(np.std(vals, ddof=1))
        empty[f"{name}_mean"] = float(np.mean(vals))
    empty["n_windows"] = float(len(window_metrics))
    return empty


def analyze_manifold_extended(embedding: np.ndarray) -> Dict[str, float]:
    """Helper."""
    empty = aggregate_all_metrics([])
    X3 = embed_umap_3d(embedding)
    if X3 is None:
        return empty
    windows = sliding_window_all_metrics(X3)
    agg = aggregate_all_metrics(windows)
    agg["umap_dim"] = float(X3.shape[1])
    return agg


def analyze_physical_only(embedding: np.ndarray) -> Dict[str, float]:
    """Helper."""
    full = analyze_manifold_extended(embedding)
    keys = [f"{m}_{s}" for m in PHYSICAL_METRIC_NAMES for s in ("std", "mean")]
    keys += ["n_windows", "umap_dim"]
    return {k: full.get(k, float("nan")) for k in keys}


def _aggregate_metric_windows(
    window_metrics: List[Dict[str, float]], names: Sequence[str]
) -> Dict[str, float]:
    empty: Dict[str, float] = {}
    for name in names:
        empty[f"{name}_std"] = float("nan")
        empty[f"{name}_mean"] = float("nan")
    empty["n_windows"] = float(len(window_metrics))
    if len(window_metrics) < 2:
        return empty
    for name in names:
        vals = np.array([w[name] for w in window_metrics], dtype=np.float64)
        empty[f"{name}_std"] = float(np.std(vals, ddof=1))
        empty[f"{name}_mean"] = float(np.mean(vals))
    return empty


def _irregularity_window_metrics(X_umap: np.ndarray) -> List[Dict[str, float]]:
    """Helper."""
    X = np.asarray(X_umap, dtype=np.float64)
    n = len(X)
    if n < SLIDING_WINDOW_EPOCHS:
        return []
    out: List[Dict[str, float]] = []
    for start in range(0, n - SLIDING_WINDOW_EPOCHS + 1, SLIDING_STEP_EPOCHS):
        chunk = X[start : start + SLIDING_WINDOW_EPOCHS]
        X_m = _log_euclidean_project(chunk)
        m = {**_compute_four_metrics(X_m), **_compute_physical_metrics(X_m)}
        out.append({"MFI": m["MFI"], "LZC": m["LZC"]})
    return [m for m in out if all(np.isfinite(v) for v in m.values())]


def analyze_irregularity_metrics(embedding: np.ndarray) -> Dict[str, float]:
    """Helper."""
    empty = _aggregate_metric_windows([], IRREGULARITY_METRIC_NAMES)
    X3 = embed_umap_3d(embedding)
    if X3 is None:
        return empty
    return analyze_irregularity_from_umap(X3)


def analyze_irregularity_from_umap(X_umap: np.ndarray) -> Dict[str, float]:
    """Helper."""
    windows = _irregularity_window_metrics(X_umap)
    return _aggregate_metric_windows(windows, IRREGULARITY_METRIC_NAMES)
