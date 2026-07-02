"""Benjamini-Hochberg FDR correction."""
from __future__ import annotations

import numpy as np


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Helper."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out

    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    filled = np.empty(n, dtype=float)
    filled[order] = q
    out[valid] = filled
    return out
