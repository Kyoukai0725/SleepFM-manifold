"""Shared LASSO utilities."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from analyze_lasso_psqi_components import _fit_lasso  # noqa: F401 re-export
from physical_dynamics_metrics import ALL_METRIC_NAMES

MANIFOLD_COLS = [f"{m}_{s}" for m in ALL_METRIC_NAMES for s in ("std", "mean")]


def run_lasso_target(
    df: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    *,
    cv_folds: int = 5,
) -> Dict:
    cols = [c for c in predictors if c in df.columns]
    sub = df.dropna(subset=[target] + cols)
    if len(sub) < len(cols) + 3:
        return {
            "target": target,
            "status": "skipped",
            "n": int(len(sub)),
            "n_features_in": len(cols),
            "reason": "complete cases too few",
            "selected_features": [],
            "selected_manifold": [],
        }
    y = sub[target].values.astype(np.float64)
    X = sub[cols].values.astype(np.float64)
    folds = cv_folds if len(sub) >= 15 else max(2, len(sub) - 1)
    res = _fit_lasso(X, y, list(cols), cv_folds=folds)
    sel_names = [x["feature"] for x in res["selected_features"]]
    res["target"] = target
    res["status"] = "ok"
    res["selected_manifold"] = [f for f in sel_names if f in MANIFOLD_COLS]
    return res


def selected_by_target(results: List[Dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        out[r["target"]] = list(r.get("selected_manifold") or [])
    return out


def md_lasso_table(results: List[Dict]) -> List[str]:
    lines = [
        "",
        "|------|---|---|------|-------|---------|-------|----------|----------|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| {r['target']} | {r.get('n', '—')} | {r.get('n_features_in', '—')} | "
                f"— | — | — | — | — | _{r.get('reason', 'skip')}_ |"
            )
            continue
        mf = ", ".join(r["selected_manifold"]) or "(none)"
        allf = ", ".join(x["feature"] for x in r["selected_features"]) or "(none)"
        lines.append(
            f"| {r['target']} | {r['n']} | {r['n_features_in']} | {r['n_selected']} | "
            f"{r['CV_R2']:+.3f} | {r['R2']:.3f} | {r['alpha']:.3f} | {mf} | {allf} |"
        )
    return lines
