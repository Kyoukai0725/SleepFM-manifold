"""DREAMT wristband epoch dataset."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_dreamt import (
    DATA_100HZ,
    NPY_WATCH,
    SAMPLES_PER_EPOCH,
    SAMPLES_PER_EPOCH_CSV,
    WEARABLE_CACHE_DIR,
    WEARABLE_COLS,
)
from preprocess_dreamt import list_psg_subjects


def _wearable_npz_path(sid: str) -> str:
    return os.path.join(WEARABLE_CACHE_DIR, f"{sid}.npz")


def _load_watch_npy(sid: str) -> Optional[np.ndarray]:
    """distill/npy/watch: (n_epochs, T, C) → (n_epochs, C, T)。"""
    path = os.path.join(NPY_WATCH, f"{sid}.npy")
    if not os.path.isfile(path):
        return None
    arr = np.load(path)
    if arr.ndim != 3:
        return None
    return np.transpose(arr, (0, 2, 1)).astype(np.float32, copy=False)


def extract_wearable_epochs(sid: str, skip_existing: bool = True) -> Optional[np.ndarray]:
    """Helper."""
    npy = _load_watch_npy(sid)
    if npy is not None:
        return npy

    out_path = _wearable_npz_path(sid)
    if skip_existing and os.path.isfile(out_path):
        return np.load(out_path)["epochs"]

    csv_path = os.path.join(DATA_100HZ, f"{sid}_PSG_df_updated.csv")
    if not os.path.isfile(csv_path):
        return None

    df = pd.read_csv(csv_path, usecols=list(WEARABLE_COLS))
    n_epochs = len(df) // SAMPLES_PER_EPOCH_CSV
    if n_epochs < 10:
        return None

    epochs = np.empty((n_epochs, len(WEARABLE_COLS), SAMPLES_PER_EPOCH_CSV), dtype=np.float32)
    for i in range(n_epochs):
        block = df.iloc[i * SAMPLES_PER_EPOCH_CSV : (i + 1) * SAMPLES_PER_EPOCH_CSV].to_numpy(dtype=np.float32)
        for c in range(block.shape[1]):
            col = block[:, c]
            mu, sd = float(col.mean()), float(col.std())
            epochs[i, c] = (col - mu) / (sd + 1e-6)
    os.makedirs(WEARABLE_CACHE_DIR, exist_ok=True)
    np.savez_compressed(out_path, epochs=epochs)
    return epochs


def build_wearable_cache(sids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Helper."""
    sids = sids or list_psg_subjects()
    out: Dict[str, np.ndarray] = {}
    for i, sid in enumerate(sids, 1):
        arr = extract_wearable_epochs(sid)
        if arr is not None:
            out[sid] = arr
        if i % 20 == 0 or i == len(sids):
            print(f"   [{i}/{len(sids)}]  {len(out)}", flush=True)
    return out


def align_teacher_wearable(
    sid: str, teacher: Dict[str, np.ndarray], wearable: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper."""
    t = teacher[sid]
    n = min(len(t), len(wearable))
    return wearable[:n], t[:n]
