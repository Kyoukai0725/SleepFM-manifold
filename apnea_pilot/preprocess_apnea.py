"""Preprocess APNEA tensors into SleepFM epoch arrays."""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly

from config_apnea import (
    APNEA_DATA_ROOT,
    APNEA_SAMPLES_PER_EPOCH,
    APNEA_SOURCE_FS,
    APNEA_TO_SLEEPFM,
    EPOCH_SEC,
    MIN_EPOCHS,
    OUTPUT_DIR,
    TARGET_FS,
)


def list_apnea_patients() -> List[str]:
    paths = sorted(glob.glob(os.path.join(APNEA_DATA_ROOT, "patient_*.pt")))
    return [os.path.basename(p).replace("patient_", "").replace(".pt", "") for p in paths]


def _resample_epoch(ch_1d: np.ndarray) -> np.ndarray:
    target_len = int(round(EPOCH_SEC * TARGET_FS))
    if len(ch_1d) != APNEA_SAMPLES_PER_EPOCH:
        ch_1d = ch_1d[:APNEA_SAMPLES_PER_EPOCH]
    out = resample_poly(ch_1d.astype(np.float64), TARGET_FS, APNEA_SOURCE_FS)
    if len(out) > target_len:
        return out[:target_len].astype(np.float32)
    if len(out) < target_len:
        return np.pad(out, (0, target_len - len(out)), mode="constant").astype(np.float32)
    return out.astype(np.float32)


def apnea_epoch_to_sleepfm(epoch_nct: np.ndarray, n_channels: int = 13) -> np.ndarray:
    """Helper."""
    target_len = int(round(EPOCH_SEC * TARGET_FS))
    mat = np.zeros((n_channels, target_len), dtype=np.float32)
    for src_i, dst_i in APNEA_TO_SLEEPFM.items():
        if src_i < epoch_nct.shape[0]:
            mat[dst_i] = _resample_epoch(epoch_nct[src_i])
    return mat


def preprocess_patient(
    patient_id: str,
    output_dir: str = OUTPUT_DIR,
    skip_existing: bool = True,
) -> Optional[str]:
    pt_path = os.path.join(APNEA_DATA_ROOT, f"patient_{patient_id}.pt")
    if not os.path.isfile(pt_path):
        return None

    out_dir = os.path.join(output_dir, "X", patient_id)
    if skip_existing and os.path.isdir(out_dir):
        n_existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
        if n_existing >= MIN_EPOCHS:
            return out_dir

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    X = data["X"].numpy() if hasattr(data["X"], "numpy") else np.asarray(data["X"])
    if X.ndim != 3 or X.shape[0] < MIN_EPOCHS:
        return None

    os.makedirs(out_dir, exist_ok=True)
    for i in range(X.shape[0]):
        mat = apnea_epoch_to_sleepfm(X[i])
        np.save(os.path.join(out_dir, f"epoch_{i:05d}.npy"), mat)
    return out_dir


def preprocess_all(output_dir: str = OUTPUT_DIR, skip_existing: bool = True) -> Tuple[List[str], List[str]]:
    ok, fail = [], []
    for pid in list_apnea_patients():
        out = preprocess_patient(pid, output_dir=output_dir, skip_existing=skip_existing)
        (ok if out else fail).append(pid)
    return ok, fail
