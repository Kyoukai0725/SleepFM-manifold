"""Load DREAMT clinical outcomes."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from config_dreamt import DATA_100HZ, EPOCH_SEC, SLEEP_STAGES, SOURCE_FS


def _psg_path(sid: str) -> str:
    return f"{DATA_100HZ}/{sid}_PSG_df_updated.csv"


def compute_sleep_outcomes(sid: str) -> Optional[Dict[str, float]]:
    """Helper."""
    path = _psg_path(sid)
    try:
        df = pd.read_csv(path, usecols=["TIMESTAMP", "Sleep_Stage"])
    except FileNotFoundError:
        return None

    if df.empty:
        return None

    spp = int(round(EPOCH_SEC * SOURCE_FS))  # 3000 @ 100Hz
    n = len(df)
    n_epochs = n // spp
    if n_epochs < 1:
        return None

    stages = df["Sleep_Stage"].iloc[::spp].iloc[:n_epochs].astype(str).values
    sleep_epochs = int(np.sum([s in SLEEP_STAGES for s in stages]))
    tst_hours = sleep_epochs * EPOCH_SEC / 3600.0

    t0 = float(df["TIMESTAMP"].iloc[0])
    t1 = float(df["TIMESTAMP"].iloc[-1])
    monitor_hours = max((t1 - t0) / 3600.0, 1e-6)
    efficiency = float(np.clip(tst_hours / monitor_hours, 0.0, 1.0))

    return {
        "TST_hours": float(tst_hours),
        "sleep_efficiency": efficiency,
        "monitor_hours": float(monitor_hours),
        "n_epochs_stage": float(n_epochs),
        "n_sleep_epochs": float(sleep_epochs),
    }
