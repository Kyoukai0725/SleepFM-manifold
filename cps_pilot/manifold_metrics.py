"""Riemannian manifold metrics via UMAP-3D and sliding windows."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def _load_umap_class():
    """Helper."""
    try:
        from umap.umap_ import UMAP

        return UMAP
    except (ImportError, AttributeError, TypeError):
        import importlib.util
        from pathlib import Path

        for base in Path(__file__).resolve().parents:
            for site in (base / "Lib" / "site-packages",):
                umap_py = site / "umap" / "umap_.py"
                if not umap_py.is_file():
                    continue
                spec = importlib.util.spec_from_file_location("_umap_impl", umap_py)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.UMAP
        import site

        for sp in site.getsitepackages():
            umap_py = Path(sp) / "umap" / "umap_.py"
            if not umap_py.is_file():
                continue
            spec = importlib.util.spec_from_file_location("_umap_impl", umap_py)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.UMAP
        raise ImportError("")


UMAP = _load_umap_class()

from config_cps import EPOCH_SEC, UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, SLIDING_STEP_MIN, SLIDING_WINDOW_MIN

METRIC_NAMES = ("LCI", "SMI", "MFI", "VTI")
PRIMARY_METRIC_NAMES = tuple(f"{m}_std" for m in METRIC_NAMES)


def _window_epochs(minutes: float) -> int:
    return max(2, int(round(minutes * 60.0 / EPOCH_SEC)))


SLIDING_WINDOW_EPOCHS = _window_epochs(SLIDING_WINDOW_MIN)
SLIDING_STEP_EPOCHS = _window_epochs(SLIDING_STEP_MIN)


def embed_umap_3d(embedding: np.ndarray, n_components: int = UMAP_N_COMPONENTS) -> Optional[np.ndarray]:
    """Helper."""
    X = np.asarray(embedding, dtype=np.float64)
    if X.ndim != 2 or len(X) < max(UMAP_N_NEIGHBORS + 1, 15):
        return None
    n_neighbors = min(UMAP_N_NEIGHBORS, len(X) - 1)
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=0,
    )
    return reducer.fit_transform(X)


def _log_euclidean_project(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n, d = X.shape
    if n < 2:
        return X
    cov = np.cov(X.T) + eps * np.eye(d)
    L = np.linalg.cholesky(cov)
    return (np.linalg.inv(L) @ X.T).T


def compute_lci(X: np.ndarray) -> float:
    if len(X) < 3:
        return float("nan")
    v1 = X[1:-1] - X[:-2]
    v2 = X[2:] - X[1:-1]
    return float(np.mean(np.linalg.norm(v2 - v1, axis=1)))


def _cluster_density_radius(X: np.ndarray, labels: np.ndarray, cluster_id: int) -> Tuple[float, float]:
    pts = X[labels == cluster_id]
    if len(pts) < 2:
        return 0.0, 1.0
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    radius = float(np.percentile(dists, 90) + 1e-8)
    density = float(len(pts) / radius)
    return density, radius


def get_largest_dense_cluster(labels: np.ndarray, X: np.ndarray) -> int:
    best_id, best_score = -1, -1.0
    for cid in np.unique(labels):
        n = int(np.sum(labels == cid))
        if n < 5:
            continue
        density, radius = _cluster_density_radius(X, labels, cid)
        score = n * density / radius
        if score > best_score:
            best_score, best_id = score, int(cid)
    return best_id if best_id >= 0 else int(np.bincount(labels).argmax())


def compute_smi(X: np.ndarray, n_clusters: int = 10) -> float:
    if len(X) < 8:
        return float("nan")
    k = min(n_clusters, max(2, len(X) // 8))
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(X)
    cid = get_largest_dense_cluster(labels, X)
    density, radius = _cluster_density_radius(X, labels, cid)
    return float(density / radius)


def compute_geometric_permutation_entropy(dists: np.ndarray, window: int = 30) -> float:
    if len(dists) < window + 2:
        return float("nan")
    entropies = []
    order = 3
    for i in range(len(dists) - window):
        w = dists[i : i + window]
        patterns = [tuple(np.argsort(w[j : j + order])) for j in range(len(w) - order + 1)]
        if not patterns:
            continue
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        p = counts / counts.sum()
        entropies.append(float(-np.sum(p * np.log(p + 1e-12))))
    return float(np.mean(entropies)) if entropies else float("nan")


def compute_mfi(X: np.ndarray, window: int = 30) -> float:
    if len(X) < 3:
        return float("nan")
    step_dists = np.linalg.norm(np.diff(X, axis=0), axis=1)
    win = min(window, max(3, len(step_dists) // 2))
    return compute_geometric_permutation_entropy(step_dists, window=win)


def compute_vti(X: np.ndarray, density_weighted: bool = True) -> float:
    """Helper."""
    if len(X) < 10:
        return float("nan")
    k = min(5, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    n = len(X)
    T = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        nbrs = idx[i, 1:]
        w = np.ones(len(nbrs))
        if density_weighted:
            w = 1.0 / (np.linalg.norm(X[nbrs] - X[i], axis=1) + 1e-8)
        T[i, nbrs] = w / w.sum()
    mags = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    if len(mags) < 2:
        return float("nan")
    return float(mags[1])


def _compute_four_metrics(X_manifold: np.ndarray) -> Dict[str, float]:
    return {
        "LCI": compute_lci(X_manifold),
        "SMI": compute_smi(X_manifold),
        "MFI": compute_mfi(X_manifold),
        "VTI": compute_vti(X_manifold, density_weighted=True),
    }


def sliding_window_metrics(
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
        m = _compute_four_metrics(X_m)
        if all(np.isfinite(v) for v in m.values()):
            out.append(m)
    return out


def _aggregate_std(window_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Helper."""
    result: Dict[str, float] = {}
    if len(window_metrics) < 2:
        for name in METRIC_NAMES:
            result[f"{name}_std"] = float("nan")
            result[f"{name}_mean"] = float("nan")
        result["n_windows"] = float(len(window_metrics))
        return result

    for name in METRIC_NAMES:
        vals = np.array([w[name] for w in window_metrics], dtype=np.float64)
        result[f"{name}_std"] = float(np.std(vals, ddof=1))
        result[f"{name}_mean"] = float(np.mean(vals))
    result["n_windows"] = float(len(window_metrics))
    return result


def analyze_embedding(embedding: np.ndarray) -> Dict[str, float]:
    """Helper."""
    empty = {f"{m}_std": np.nan for m in METRIC_NAMES}
    empty.update({f"{m}_mean": np.nan for m in METRIC_NAMES})
    empty["n_windows"] = 0.0

    X3 = embed_umap_3d(embedding)
    if X3 is None:
        return empty

    windows = sliding_window_metrics(X3)
    agg = _aggregate_std(windows)
    agg["umap_dim"] = float(X3.shape[1])
    return agg
