"""CPS clinical outcome helpers: AHI, ARI, and sleep duration."""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from env_paths import CPS_DATA, CPS_PSQI_OUTPUT, CPS_ROOT  # noqa: E402

CPS_ROOT = str(CPS_ROOT)
CPS_DATA = str(CPS_DATA)
OUTPUT_DIR = str(CPS_PSQI_OUTPUT)
_metrics = sorted(glob.glob(os.path.join(OUTPUT_DIR, "metrics_umap_sliding_*.csv")))
METRICS_CSV = _metrics[-1] if _metrics else os.path.join(OUTPUT_DIR, "metrics_umap_sliding.csv")
SKIP_SUBJECTS = frozenset({
    "ZwojnSyDEhD6s8ZERY8AoDWL0cnBH7BU",
    "wuxfEVb6iJ7GghVUjN5eKTO0VsulXaOg",
})

def load_valid_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV)
    df = df[~df["subject_id"].isin(SKIP_SUBJECTS)]
    df = df[(df["n_epochs"] > 0) & df["LCI_std"].notna()]
    return df

if CPS_ROOT not in sys.path:
    sys.path.insert(0, CPS_ROOT)

from load import (  # noqa: E402
    count_cps_scorer_apnoe_hypopnoe,
    compute_cps_scorer_ahi,
    sleep_duration_hours_tst_schlafprofil,
)
from utils import read_event_file_as_list  # noqa: E402

from manifold_metrics import PRIMARY_METRIC_NAMES

MANIFOLD_COLS = list(PRIMARY_METRIC_NAMES)
AROUSAL_FILE = "Klassifizierte Arousal.txt"


def _sleep_hours(base_path: str, sample_id: str) -> float:
    """Helper."""
    try:
        from create_statistics import sleep_duration  # noqa: WPS433

        if sample_id in sleep_duration:
            return float(sleep_duration[sample_id])
    except Exception:
        pass
    return sleep_duration_hours_tst_schlafprofil(base_path, sample_id)


def compute_ari(
    base_path: str,
    sample_id: str,
    *,
    eeg_only: bool = False,
) -> Optional[Dict[str, float]]:
    """Helper."""
    path = os.path.join(base_path, sample_id, "PSG", "Analysedaten", AROUSAL_FILE)
    try:
        rows, typ, _ = read_event_file_as_list(path)
    except FileNotFoundError:
        return None
    if typ != "Impuls" or not rows:
        return None

    if eeg_only:
        n = sum(
            1
            for row in rows
            if isinstance(row, (list, tuple)) and len(row) >= 4 and "(EEG)" in str(row[3])
        )
    else:
        n = len(rows)

    sleep_h = _sleep_hours(base_path, sample_id)
    if not np.isfinite(sleep_h) or sleep_h <= 0:
        return None

    ari = float(n / sleep_h)
    return {
        "n_arousals": float(n),
        "sleep_hours": float(sleep_h),
        "ARI": ari,
    }


def compute_ahi_flow(base_path: str, sample_id: str) -> float:
    path = os.path.join(base_path, sample_id, "PSG", "Analysedaten", "Flow Events.txt")
    try:
        rows, _, _ = read_event_file_as_list(path)
    except FileNotFoundError:
        return float("nan")
    a, h = count_cps_scorer_apnoe_hypopnoe(rows or [])
    sh = _sleep_hours(base_path, sample_id)
    if not np.isfinite(sh) or sh <= 0:
        return float("nan")
    return float((a + h) / sh)


def _cv_r2(X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    if len(y) < folds + 2:
        return float("nan")
    Xs = StandardScaler().fit_transform(X)
    scores = cross_val_score(LinearRegression(), Xs, y, cv=folds, scoring="r2")
    return float(np.mean(scores))


def _fit_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    Xs = StandardScaler().fit_transform(X)
    model = LinearRegression().fit(Xs, y)
    pred = model.predict(Xs)
    r2 = r2_score(y, pred)
    n, p = X.shape
    adj = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    return float(r2), float(adj)


def run_analysis(ari_target: str = "ARI") -> pd.DataFrame:
    metrics = load_valid_metrics()
    rows = []
    for sid in metrics["subject_id"]:
        ari_all = compute_ari(CPS_DATA, sid, eeg_only=False)
        ari_eeg = compute_ari(CPS_DATA, sid, eeg_only=True)
        ahi_info = compute_cps_scorer_ahi(CPS_DATA, sid)
        row = {
            "subject_id": sid,
            "ahi_cps_scorer": ahi_info.get("ahi_cps_scorer", np.nan),
            "ahi_flow_only": compute_ahi_flow(CPS_DATA, sid),
        }
        if ari_all:
            row.update({f"{k}_all": v for k, v in ari_all.items()})
        if ari_eeg:
            row.update({f"{k}_eeg": v for k, v in ari_eeg.items()})
        rows.append(row)

    clinical = pd.DataFrame(rows)
    df = metrics.merge(clinical, on="subject_id", how="left")
    df = df.dropna(subset=[ari_target])

    predictors = MANIFOLD_COLS + ["ahi_cps_scorer", "ahi_flow_only"]
    corr_rows = []
    for pred in predictors:
        sub = df[[ari_target, pred]].dropna()
        if len(sub) < 8:
            rho, p = np.nan, np.nan
        else:
            rho, p = stats.spearmanr(sub[ari_target], sub[pred])
        corr_rows.append({"target": ari_target, "predictor": pred, "spearman_r": rho, "p_value": p, "n": len(sub)})
    corr_df = pd.DataFrame(corr_rows)

    y = df[ari_target].values
    valid = np.isfinite(y)
    for c in MANIFOLD_COLS + ["ahi_flow_only"]:
        valid &= np.isfinite(df[c].values)
    dfv = df.loc[valid].copy()
    y = dfv[ari_target].values

    models = {
        "AHI_flow_only": dfv[["ahi_flow_only"]].values,
        "AHI_cps_scorer": dfv[["ahi_cps_scorer"]].values,
        "Manifold_only": dfv[MANIFOLD_COLS].values,
        "Manifold_plus_AHI_flow": dfv[["ahi_flow_only"] + MANIFOLD_COLS].values,
    }
    reg_rows = []
    for name, X in models.items():
        sub_valid = np.all(np.isfinite(X), axis=1)
        r2, adj = _fit_r2(X[sub_valid], y[sub_valid])
        cv = _cv_r2(X[sub_valid], y[sub_valid])
        reg_rows.append({"model": name, "R2": r2, "Adj_R2": adj, "CV_R2": cv, "n": int(sub_valid.sum())})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = os.path.join(OUTPUT_DIR, f"ari_manifold_merged_{ts}.csv")
    corr_path = os.path.join(OUTPUT_DIR, f"ari_correlations_{ts}.csv")
    reg_path = os.path.join(OUTPUT_DIR, f"ari_regression_compare_{ts}.csv")
    df.to_csv(merged_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    pd.DataFrame(reg_rows).to_csv(reg_path, index=False)

    summary = {
        "n_subjects": int(len(dfv)),
        f"{ari_target}_mean": float(dfv[ari_target].mean()),
        f"{ari_target}_std": float(dfv[ari_target].std()),
        "regression_comparison": reg_rows,
        "correlations": corr_rows,
        "files": {"merged": merged_path, "correlations": corr_path, "regression": reg_path},
    }
    summary_path = os.path.join(OUTPUT_DIR, f"ari_analysis_summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f": {len(dfv)}  |  {ari_target} ={dfv[ari_target].mean():.2f} ± {dfv[ari_target].std():.2f} /h")
    print("\n=== Spearman （ ARI）===")
    for row in corr_rows:
        sig = "*" if row["p_value"] < 0.05 else ("**" if row["p_value"] < 0.01 else "")
        print(f"  {row['predictor']:16s}  r={row['spearman_r']:+.3f}  p={row['p_value']:.4f}{sig}")

    print("\n=== Regression (predict ARI, higher R2 is better) ===")
    for r in reg_rows:
        print(f"  {r['model']:22s}  R2={r['R2']:.3f}  Adj_R2={r['Adj_R2']:.3f}  CV_R2={r['CV_R2']:.3f}")

    print(f"\n: {summary_path}")
    return df


if __name__ == "__main__":
    print(">>> ARI（）")
    run_analysis("ARI_all")
    print("\n>>> ARI_EEG（ (EEG) ）")
    run_analysis("ARI_eeg")
