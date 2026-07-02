"""Preprocess CPS WFDB recordings into SleepFM epoch arrays."""
from __future__ import annotations

import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample_poly

_CPS_ROOT = r"E:\psg_dataset\CPS"
if _CPS_ROOT not in sys.path:
    sys.path.insert(0, _CPS_ROOT)

from load import get_recording_start_sec  # noqa: E402
from utils import read_event_file_as_list  # noqa: E402

from config_cps import (
    ALL_CHANNELS,
    BANDPASS,
    CPS_CHANNEL_CANDIDATES,
    CPS_DATA_ROOT,
    CPS_STAGE_MAP,
    EPOCH_SEC,
    TARGET_FS,
)


def _parse_clock(s: str) -> float:
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) >= 3:
        h, m = int(parts[0]), int(parts[1])
        sec = float(".".join(parts[2:]))
        return h * 3600 + m * 60 + sec
    return 0.0


def _clock_to_elapsed(clock_sec: float, ref_sec: float) -> float:
    if clock_sec >= ref_sec:
        return clock_sec - ref_sec
    return clock_sec + (86400.0 - ref_sec)


def parse_sleep_stages(base_path: str, sample_id: str) -> List[Tuple[float, str]]:
    """Helper."""
    path = os.path.join(base_path, sample_id, "PSG", "Analysedaten", "Schlafprofil.txt")
    rows, typ, _ = read_event_file_as_list(path)
    if typ != "Discret" or not rows:
        return []

    ref_sec = get_recording_start_sec(base_path, sample_id)
    stages: List[Tuple[float, str]] = []
    for row in rows:
        if len(row) < 2:
            continue
        stage_raw = str(row[1]).strip()
        if stage_raw in ("A", "Artefakt"):
            continue
        elapsed = _clock_to_elapsed(_parse_clock(str(row[0])), ref_sec)
        mapped = CPS_STAGE_MAP.get(stage_raw)
        if mapped:
            stages.append((elapsed, mapped))
    return stages


def _bandpass_1d(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    lo, hi = low / nyq, high / nyq
    if hi >= 1.0:
        hi = 0.99
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x.astype(np.float64))


def load_patient_channels(base_path: str, sample_id: str) -> Tuple[Dict[str, np.ndarray], float]:
    psg_path = os.path.join(base_path, sample_id, "PSG", sample_id)
    record = wfdb.rdrecord(psg_path)
    fs = float(getattr(record, "fs", TARGET_FS))
    sig = np.transpose(record.p_signal)
    ch_map = {str(n).strip(): np.asarray(sig[i], dtype=np.float64) for i, n in enumerate(record.sig_name)}
    return ch_map, fs


def _resample_channel(x: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    if abs(fs_orig - fs_target) < 1e-3:
        return x
    up = int(round(fs_target))
    down = int(round(fs_orig))
    if up <= 0 or down <= 0:
        return resample_poly(x, int(fs_target), int(fs_orig))
    return resample_poly(x, up, down)


def _pick_channel(ch_map: Dict[str, np.ndarray], candidates: List[str]) -> Optional[np.ndarray]:
    for name in candidates:
        if name in ch_map:
            return ch_map[name]
    return None


def build_sleepfm_epoch_matrix(
    ch_map: Dict[str, np.ndarray],
    fs_orig: float,
    start_sec: float,
    duration_sec: float = EPOCH_SEC,
) -> np.ndarray:
    """Helper."""
    start_samp = int(round(start_sec * fs_orig))
    end_samp = int(round((start_sec + duration_sec) * fs_orig))
    rows = []

    eeg_names = {"F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2"}

    for sf_name in ALL_CHANNELS:
        candidates = CPS_CHANNEL_CANDIDATES[sf_name]
        raw = _pick_channel(ch_map, candidates)
        if raw is None:
            target_len = int(round(duration_sec * TARGET_FS))
            rows.append(np.zeros(target_len, dtype=np.float64))
            continue

        seg = raw[start_samp:end_samp]
        seg = _resample_channel(seg, fs_orig, TARGET_FS)
        if sf_name in eeg_names:
            seg = _bandpass_1d(seg, TARGET_FS, BANDPASS[0], BANDPASS[1])
        rows.append(seg)

    target_len = int(round(duration_sec * TARGET_FS))
    rows = [
        r if len(r) == target_len else np.pad(r, (0, max(0, target_len - len(r))), mode="constant")[:target_len]
        for r in rows
    ]
    mat = np.stack(rows, axis=0)
    return mat.astype(np.float32)


def preprocess_one_subject(
    sample_id: str,
    output_dir: str,
    base_path: str = CPS_DATA_ROOT,
) -> Optional[str]:
    """Helper."""
    stages = parse_sleep_stages(base_path, sample_id)
    if not stages:
        return None

    ch_map, fs_orig = load_patient_channels(base_path, sample_id)
    path_x = os.path.join(output_dir, "X", sample_id)
    path_y = os.path.join(output_dir, "Y", f"{sample_id}.pickle")
    os.makedirs(path_x, exist_ok=True)
    os.makedirs(os.path.dirname(path_y), exist_ok=True)

    labels: Dict[str, str] = {}
    for idx, (elapsed, stage) in enumerate(stages):
        epoch = build_sleepfm_epoch_matrix(ch_map, fs_orig, elapsed, EPOCH_SEC)
        if epoch.shape[1] == 0:
            continue
        fname = f"{sample_id}_{idx}.npy"
        np.save(os.path.join(path_x, fname), epoch)
        labels[fname] = stage

    if not labels:
        return None

    with open(path_y, "wb") as f:
        pickle.dump(labels, f)
    return path_x


def preprocess_subjects(
    sample_ids: List[str],
    output_dir: str,
    base_path: str = CPS_DATA_ROOT,
) -> List[str]:
    ok = []
    for sid in sample_ids:
        try:
            out = preprocess_one_subject(sid, output_dir, base_path)
            if out:
                ok.append(sid)
        except Exception as exc:
            print(f"[WARN] {sid}: {exc}")
    return ok
