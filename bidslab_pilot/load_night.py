"""Load Bidslab watch and EEG labels."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import scipy.io as sio

from config_bidslab import EPOCH_SEC

_EASTERN = pytz.timezone("US/Eastern")


def list_subjects(root: Path) -> List[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("bidslab"))


def list_nights(subject_dir: Path) -> List[int]:
    out = []
    for p in subject_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            out.append(int(p.name))
    return sorted(out)


def parse_rec_start_unix(val) -> float:
    if isinstance(val, (float, np.floating, int, np.integer)):
        return float(val)
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            naive = dt.datetime.strptime(s, fmt)
            return _EASTERN.localize(naive).timestamp()
        except ValueError:
            continue
    ts = pd.to_datetime(s, utc=False)
    if ts.tzinfo is None:
        ts = _EASTERN.localize(ts.to_pydatetime())
        return float(ts.timestamp())
    return float(ts.timestamp())


def load_labels(labels_path: Path) -> Tuple[float, np.ndarray, np.ndarray]:
    d = sio.loadmat(str(labels_path))
    rec = parse_rec_start_unix(d["recStart"].ravel()[0])
    dreem = np.asarray(d["dreem_label"]).ravel().astype(np.int64)
    expert = np.asarray(d["expert_label"]).ravel().astype(np.int64)
    return rec, dreem, expert


def load_motion(motion_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(motion_path, usecols=["Timestamp", "x", "y", "z"])
    t = df["Timestamp"].to_numpy(dtype=np.float64)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return t, xyz


def load_hr(hr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(hr_path, header=None, names=["timestamp", "bpm"])
    t = df["timestamp"].to_numpy(dtype=np.float64)
    bpm = df["bpm"].to_numpy(dtype=np.float64)
    return t, bpm


def _resample_1d(x: np.ndarray, n_out: int) -> np.ndarray:
    n_in = x.size
    if n_in == n_out:
        return x.astype(np.float32)
    if n_in <= 1:
        return np.full((n_out,), float(x[0]) if n_in else 0.0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, n_in, dtype=np.float64)
    xnew = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
    return np.interp(xnew, xp, x).astype(np.float32)


def _slice_resample(
    t: np.ndarray,
    y: np.ndarray,
    t0: float,
    t1: float,
    n_out: int,
) -> np.ndarray:
    mask = (t >= t0) & (t < t1)
    if not np.any(mask):
        return np.zeros(n_out, dtype=np.float32)
    seg_t = t[mask]
    seg_y = y[mask]
    order = np.argsort(seg_t)
    seg_t, seg_y = seg_t[order], seg_y[order]
    rel = (seg_t - t0) / max(t1 - t0, 1e-6)
    out = np.interp(np.linspace(0, 1, n_out), rel, seg_y).astype(np.float32)
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = float(x.mean()), float(x.std())
    return ((x - mu) / (sd + 1e-8)).astype(np.float32)


def build_epoch_watch(
    rec_start: float,
    expert_label: np.ndarray,
    motion_t: np.ndarray,
    motion_xyz: np.ndarray,
    hr_t: np.ndarray,
    hr_bpm: np.ndarray,
    samples_per_epoch: int,
) -> np.ndarray:
    """Helper."""
    n_epochs = len(expert_label)
    mag = np.linalg.norm(motion_xyz, axis=1)
    epochs = []
    for i in range(n_epochs):
        t0 = rec_start + i * EPOCH_SEC
        t1 = t0 + EPOCH_SEC
        ihr = _slice_resample(hr_t, hr_bpm, t0, t1, samples_per_epoch)
        acc = _slice_resample(motion_t, mag, t0, t1, samples_per_epoch)
        dihr = np.diff(ihr, prepend=ihr[0])
        ch = np.stack([_zscore(ihr), _zscore(acc), _zscore(dihr)], axis=0)
        epochs.append(ch)
    return np.stack(epochs, axis=0)


def epoch_watch_to_embedding(watch: np.ndarray, pool: int = 64) -> np.ndarray:
    """Helper."""
    e, c, t = watch.shape
    if t % pool != 0:
        t_new = (t // pool) * pool
        if t_new < pool:
            t_new = pool
        out = np.empty((e, c, t_new), dtype=np.float32)
        for i in range(e):
            for j in range(c):
                out[i, j] = _resample_1d(watch[i, j], t_new)
        watch = out
        t = t_new
    bin_w = t // pool
    x = watch.reshape(e, c, pool, bin_w).mean(axis=3)
    return x.reshape(e, c * pool).astype(np.float32)
