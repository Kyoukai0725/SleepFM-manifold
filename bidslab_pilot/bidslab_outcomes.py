"""Compute Bidslab TST and sleep efficiency."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from config_bidslab import EPOCH_SEC, SLEEP_STAGES, UNKNOWN_STAGE


def compute_outcomes(expert_label: np.ndarray) -> Optional[Dict[str, float]]:
    stages = np.asarray(expert_label, dtype=np.int64).ravel()
    if stages.size < 10:
        return None
    valid = stages != UNKNOWN_STAGE
    if valid.sum() < 10:
        return None
    stages = stages[valid]
    n_epochs = len(stages)
    sleep_epochs = int(np.sum(np.isin(stages, list(SLEEP_STAGES))))
    tst_hours = sleep_epochs * EPOCH_SEC / 3600.0
    monitor_hours = n_epochs * EPOCH_SEC / 3600.0
    if monitor_hours < 1e-6:
        return None
    return {
        "TST_hours": float(tst_hours),
        "sleep_efficiency": float(np.clip(tst_hours / monitor_hours, 0.0, 1.0)),
        "n_epochs": float(n_epochs),
        "n_sleep_epochs": float(sleep_epochs),
        "n_wake_epochs": float(np.sum(stages == 0)),
    }
