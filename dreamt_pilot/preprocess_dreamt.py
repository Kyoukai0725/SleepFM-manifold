"""DREAMT 100Hz PSG → SleepFM 30s epoch (.npy)。"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

from config_dreamt import (
    ALL_CHANNELS,
    BANDPASS,
    DREAMT_CHANNEL_CANDIDATES,
    DATA_100HZ,
    EPOCH_SEC,
    READ_COLS,
    SOURCE_FS,
    TARGET_FS,
)

EEG_NAMES = {"F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2"}
SAMPLES_PER_EPOCH = int(round(EPOCH_SEC * SOURCE_FS))


def _bandpass_1d(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    lo, hi = low / nyq, high / nyq
    if hi >= 1.0:
        hi = 0.99
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x.astype(np.float64))


def _resample_channel(x: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    if abs(fs_orig - fs_target) < 1e-3:
        return x
    up, down = int(round(fs_target)), int(round(fs_orig))
    return resample_poly(x, up, down)


def _pick_col(data: Dict[str, np.ndarray], candidates: List[str]) -> Optional[np.ndarray]:
    for name in candidates:
        if name in data:
            return data[name]
    return None


def build_epoch_matrix(ch_data: Dict[str, np.ndarray], epoch_idx: int) -> np.ndarray:
    start = epoch_idx * SAMPLES_PER_EPOCH
    end = start + SAMPLES_PER_EPOCH
    target_len = int(round(EPOCH_SEC * TARGET_FS))
    rows = []
    for sf_name in ALL_CHANNELS:
        raw = _pick_col(ch_data, DREAMT_CHANNEL_CANDIDATES[sf_name])
        if raw is None:
            rows.append(np.zeros(target_len, dtype=np.float64))
            continue
        seg = raw[start:end]
        if len(seg) < SAMPLES_PER_EPOCH:
            seg = np.pad(seg, (0, SAMPLES_PER_EPOCH - len(seg)), mode="constant")
        seg = _resample_channel(seg, SOURCE_FS, TARGET_FS)
        if sf_name in EEG_NAMES:
            seg = _bandpass_1d(seg, TARGET_FS, BANDPASS[0], BANDPASS[1])
        if len(seg) != target_len:
            seg = seg[:target_len] if len(seg) > target_len else np.pad(seg, (0, target_len - len(seg)))
        rows.append(seg)
    return np.stack(rows, axis=0).astype(np.float32)


def preprocess_subject(sid: str, output_dir: str, skip_existing: bool = True) -> Optional[str]:
    out_dir = os.path.join(output_dir, "X", sid)
    if skip_existing and os.path.isdir(out_dir):
        n_existing = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
        if n_existing >= 10:
            return out_dir

    csv_path = os.path.join(DATA_100HZ, f"{sid}_PSG_df_updated.csv")
    if not os.path.isfile(csv_path):
        return None

    df = pd.read_csv(csv_path, usecols=READ_COLS)
    ch_data = {
        c: df[c].to_numpy(dtype=np.float64)
        for c in df.columns
        if c not in ("TIMESTAMP", "Sleep_Stage")
    }
    n_epochs = len(df) // SAMPLES_PER_EPOCH
    if n_epochs < 10:
        return None

    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_epochs):
        mat = build_epoch_matrix(ch_data, i)
        np.save(os.path.join(out_dir, f"epoch_{i:05d}.npy"), mat)
    return out_dir


def list_psg_subjects() -> List[str]:
    sids = []
    for fn in os.listdir(DATA_100HZ):
        if fn.endswith("_PSG_df_updated.csv"):
            sids.append(fn.split("_")[0])
    return sorted(sids)
