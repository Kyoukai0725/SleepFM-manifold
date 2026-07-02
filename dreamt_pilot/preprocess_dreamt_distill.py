"""Preprocess DREAMT PSG and wristband signals for distillation."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

_DISTILLSLEEP = r"E:\psg_dataset\watch\DistillSleep-main"
_PREPROCESS = os.path.join(_DISTILLSLEEP, "Data", "preprocess")
if os.path.isdir(_PREPROCESS) and _PREPROCESS not in sys.path:
    sys.path.insert(0, _PREPROCESS)

from filtering_utils import butter_bandpass_filter  # noqa: E402

from distill_config import (  # noqa: E402
    DATA_100HZ,
    NPY_ANN,
    NPY_PSG,
    NPY_WATCH,
    PSG_COLS,
    SAMPLES_PSG,
    SAMPLES_SRC,
    SAMPLES_WATCH,
    SOURCE_FS,
    STAGE_TO_LABEL,
    WATCH_COLS,
)


def _zscore_epoch(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float64)
    mu = out.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, keepdims=True) + 1e-8
    return ((out - mu) / sd).astype(np.float32)


def _resample_epoch(x: np.ndarray, n_out: int) -> np.ndarray:
    n_in = x.shape[0]
    if n_in == n_out:
        return x.astype(np.float32)
    return resample_poly(x, n_out, n_in).astype(np.float32)


def _watch_three_channels(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray, bvp: np.ndarray, eda: np.ndarray) -> np.ndarray:
    acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    return np.stack([bvp, acc_mag, eda], axis=1)


def _process_psg_epoch(seg: np.ndarray) -> np.ndarray:
    eeg = butter_bandpass_filter(seg[:, 0], 0.5, 30.0, SOURCE_FS)
    out = seg.copy()
    out[:, 0] = eeg
    out = _resample_epoch(out, SAMPLES_PSG)
    return _zscore_epoch(out)


def _process_watch_epoch(seg: np.ndarray) -> np.ndarray:
    out = _resample_epoch(seg, SAMPLES_WATCH)
    return _zscore_epoch(out)


def list_subjects() -> List[str]:
    sids = []
    for fn in os.listdir(DATA_100HZ):
        if fn.endswith("_PSG_df_updated.csv"):
            sids.append(fn.split("_")[0])
    return sorted(sids)


def preprocess_subject(sid: str, skip_existing: bool = True) -> Optional[Tuple[int, int]]:
    out_psg = os.path.join(NPY_PSG, f"{sid}.npy")
    out_watch = os.path.join(NPY_WATCH, f"{sid}.npy")
    out_ann = os.path.join(NPY_ANN, f"{sid}.npy")
    if skip_existing and all(os.path.isfile(p) for p in (out_psg, out_watch, out_ann)):
        n = np.load(out_ann).shape[0]
        return n, n

    csv_path = os.path.join(DATA_100HZ, f"{sid}_PSG_df_updated.csv")
    cols = ["Sleep_Stage"] + PSG_COLS + WATCH_COLS
    df = pd.read_csv(csv_path, usecols=cols)

    n_epochs = len(df) // SAMPLES_SRC
    if n_epochs < 10:
        return None

    psg_epochs, watch_epochs, labels = [], [], []
    for i in range(n_epochs):
        sl = slice(i * SAMPLES_SRC, (i + 1) * SAMPLES_SRC)
        stage = str(df["Sleep_Stage"].iloc[i * SAMPLES_SRC]).strip()
        if stage not in STAGE_TO_LABEL:
            continue
        psg_seg = df[PSG_COLS].iloc[sl].to_numpy(dtype=np.float64)
        bvp = df["BVP"].iloc[sl].to_numpy(dtype=np.float64)
        eda = df["EDA"].iloc[sl].to_numpy(dtype=np.float64)
        ax = df["ACC_X"].iloc[sl].to_numpy(dtype=np.float64)
        ay = df["ACC_Y"].iloc[sl].to_numpy(dtype=np.float64)
        az = df["ACC_Z"].iloc[sl].to_numpy(dtype=np.float64)
        watch_seg = _watch_three_channels(ax, ay, az, bvp, eda)

        psg_epochs.append(_process_psg_epoch(psg_seg))
        watch_epochs.append(_process_watch_epoch(watch_seg))
        labels.append(STAGE_TO_LABEL[stage])

    if len(labels) < 10:
        return None

    os.makedirs(NPY_PSG, exist_ok=True)
    os.makedirs(NPY_WATCH, exist_ok=True)
    os.makedirs(NPY_ANN, exist_ok=True)
    np.save(out_psg, np.stack(psg_epochs, axis=0))
    np.save(out_watch, np.stack(watch_epochs, axis=0))
    np.save(out_ann, np.array(labels, dtype=np.int64))
    return len(labels), len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_subjects", type=int, default=0)
    args = parser.parse_args()

    sids = list_subjects()
    if args.max_subjects > 0:
        sids = sids[: args.max_subjects]

    ok = 0
    for i, sid in enumerate(sids, 1):
        r = preprocess_subject(sid)
        if r:
            ok += 1
        if i % 10 == 0 or i == len(sids):
            print(f"  [{i}/{len(sids)}]  {ok}")
    print(f": {ok}/{len(sids)}")


if __name__ == "__main__":
    main()
