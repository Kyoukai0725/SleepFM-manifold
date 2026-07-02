"""Fixed UMAP frame with KNN projection."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as exc:
    raise ImportError("pip install umap-learn") from exc

from config_bidslab import UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_RANDOM_STATE

import sys

_CPS = Path(__file__).resolve().parent.parent / "cps_pilot"
if str(_CPS) not in sys.path:
    sys.path.insert(0, str(_CPS))

from physical_dynamics_metrics import (  # noqa: E402
    ALL_METRIC_NAMES,
    aggregate_all_metrics,
    sliding_window_all_metrics,
)

FrameType = Union[Dict[str, Any], Any]


def _min_samples_for_umap(n: int) -> bool:
    return n >= max(UMAP_N_NEIGHBORS + 1, 15)


def _clean_embeddings(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fit_umap_frame(embeddings: np.ndarray) -> Optional[Dict[str, Any]]:
    """Helper."""
    x = _clean_embeddings(embeddings)
    if not _min_samples_for_umap(len(x)):
        return None
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    nn = min(UMAP_N_NEIGHBORS, len(x) - 1)
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=nn,
        min_dist=0.1,
        metric="euclidean",
        random_state=UMAP_RANDOM_STATE,
    )
    reducer.fit(x_scaled)
    train_coords = np.asarray(reducer.embedding_, dtype=np.float64).copy()
    projector = KNeighborsRegressor(n_neighbors=min(15, len(x) - 1), weights="distance")
    projector.fit(x_scaled, train_coords)
    return {
        "scaler": scaler,
        "reducer": reducer,
        "projector": projector,
        "train_emb": x.copy(),
        "train_coords": train_coords,
    }


def _as_frame(obj: FrameType) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict) and "reducer" in obj:
        return obj
    return {"scaler": None, "reducer": obj, "projector": None, "train_emb": None, "train_coords": None}


def _impute_nan_coords(
    x_scaled: np.ndarray,
    x3: np.ndarray,
    frame: Dict[str, Any],
) -> np.ndarray:
    out = np.asarray(x3, dtype=np.float64).copy()
    bad = ~np.isfinite(out).all(axis=1)
    if not bad.any():
        return out

    train_emb = frame.get("train_emb")
    train_coords = frame.get("train_coords")
    if train_emb is None or train_coords is None:
        good = np.isfinite(out).all(axis=1)
        med = np.nanmedian(out[good], axis=0) if good.any() else np.zeros(out.shape[1])
        out[bad] = med
        return out

    scaler = frame.get("scaler")
    ref_x = train_emb if scaler is None else scaler.transform(train_emb)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(ref_x)
    _, idx = nn.kneighbors(x_scaled[bad])
    out[bad] = train_coords[idx.ravel()]
    return out


def _project_coords(x_scaled: np.ndarray, frame: Dict[str, Any]) -> np.ndarray:
    projector = frame.get("projector")
    if projector is not None:
        return np.asarray(projector.predict(x_scaled), dtype=np.float64)

    reducer = frame["reducer"]
    x3 = reducer.transform(x_scaled)
    return _impute_nan_coords(x_scaled, x3, frame)


def transform_umap(frame: FrameType, embeddings: np.ndarray) -> Optional[np.ndarray]:
    pack = _as_frame(frame)
    x = _clean_embeddings(embeddings)
    if pack is None or len(x) < 2:
        return None
    scaler = pack.get("scaler")
    x_in = scaler.transform(x) if scaler is not None else x

    if pack.get("projector") is not None:
        return _project_coords(x_in, pack)

    x3 = _impute_nan_coords(x_in, pack["reducer"].transform(x_in), pack)
    return x3


def metrics_from_umap_coords(X3: np.ndarray) -> Dict[str, float]:
    empty = aggregate_all_metrics([])
    if X3 is None or len(X3) < 2:
        return empty
    x = np.asarray(X3, dtype=np.float64)
    if not np.isfinite(x).all():
        return empty
    windows = sliding_window_all_metrics(x)
    agg = aggregate_all_metrics(windows)
    agg["umap_dim"] = float(x.shape[1])
    return agg


def analyze_fixed_frame(
    embeddings: np.ndarray,
    frame: FrameType,
    *,
    refit_nightly: bool = False,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """Helper."""
    if refit_nightly:
        from manifold_metrics import embed_umap_3d

        x3 = embed_umap_3d(embeddings)
    else:
        x3 = transform_umap(frame, embeddings)
    return metrics_from_umap_coords(x3), x3


def save_frame(path: Path, frame: FrameType) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(frame, f)


def load_frame(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("rb") as f:
        obj = pickle.load(f)
    return _as_frame(obj)


def metric_keys() -> list[str]:
    return [f"{m}_{s}" for m in ALL_METRIC_NAMES for s in ("std", "mean")]
