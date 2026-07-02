"""Rank-based partial correlation helpers."""
from __future__ import annotations

import numpy as np
from scipy import stats


def rank_residual(v: np.ndarray, C: np.ndarray) -> np.ndarray:
    r = stats.rankdata(v)
    X = np.column_stack([np.ones(len(C)), C])
    beta, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
    return r - X @ beta


def partial_spearman(y: np.ndarray, x: np.ndarray, C: np.ndarray) -> tuple[float, float]:
    yr = rank_residual(y, C)
    xr = rank_residual(x, C)
    r, p = stats.pearsonr(yr, xr)
    return float(r), float(p)
