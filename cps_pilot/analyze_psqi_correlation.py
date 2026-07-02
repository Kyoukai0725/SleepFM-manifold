"""PSQI scoring helpers."""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

CPS_ROOT = r"E:\psg_dataset\CPS"
CPS_DATA = os.path.join(CPS_ROOT, "data")
METRICS_CSV = r"E:\psg_dataset\SleepFM\output\cps_psqi\metrics_umap_sliding_20260608_091637.csv"
OUTPUT_DIR = r"E:\psg_dataset\SleepFM\output\cps_psqi"
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
from load import compute_cps_scorer_ahi  # noqa: E402

from manifold_metrics import PRIMARY_METRIC_NAMES

MANIFOLD_COLS = list(PRIMARY_METRIC_NAMES)
FREQ_SCORE = {
    "während der letzten vier wochen nicht": 0,
    "weniger als einmal pro woche": 1,
    "einmal oder zweimal pro woche": 2,
    "dreimal oder häufiger pro woche": 3,
}
QUALITY_SCORE = {
    "sehr gut": 0,
    "ziemlich gut": 1,
    "ziemlich schlecht": 2,
    "sehr schlecht": 3,
}
PROBLEM_SCORE = {
    "keine probleme": 0,
    "kaum probleme": 1,
    "etwas probleme": 2,
    "große probleme": 3,
    "grosse probleme": 3,
}


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")


def _load_yaml(path: str) -> Dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError:
            return {}


def _parse_time_minutes(s: str) -> Optional[float]:
    s = str(s).strip()
    if "-" in s and ":" not in s.split("-")[0]:
        parts = re.split(r"\s*-\s*", s)
        vals = []
        for p in parts:
            m = re.search(r"(\d+(?:[.,]\d+)?)", p)
            if m:
                vals.append(float(m.group(1).replace(",", ".")))
        if vals:
            return float(np.mean(vals))
    if "-" in s and ":" in s:
        parts = re.split(r"\s*-\s*", s)
        mins = [_parse_time_minutes(p) for p in parts]
        mins = [m for m in mins if m is not None]
        if mins:
            return float(np.mean(mins))
    m = re.match(r"(\d{1,2}):(\d{2})", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*min", _norm(s))
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _parse_sleep_hours(s: str) -> Optional[float]:
    s = _norm(s)
    # 8-9h / 6-7h
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)\s*h", s)
    if m:
        return (float(m.group(1).replace(",", ".")) + float(m.group(2).replace(",", "."))) / 2.0
    if "7-8" in s or ">7" in s:
        return 7.5
    if "6-7" in s:
        return 6.5
    if "5-6" in s:
        return 5.5
    if "<5" in s or "unter 5" in s:
        return 4.5
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*h", s)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _freq_score(val: Any) -> Optional[int]:
    key = _norm(val)
    for k, v in FREQ_SCORE.items():
        nk = _norm(k)
        if nk in key or key in nk:
            return v
    return None


def _component_latency(q2: Any, q5a: Any) -> Optional[int]:
    mins = _parse_time_minutes(str(q2)) if q2 is not None else None
    f5a = _freq_score(q5a)
    if mins is None:
        return None
    if mins <= 15:
        s2 = 0
    elif mins <= 30:
        s2 = 1
    elif mins <= 60:
        s2 = 2
    else:
        s2 = 3
    if f5a is None:
        return s2
    total = s2 + f5a
    if total == 0:
        return 0
    if total <= 2:
        return 1
    if total <= 4:
        return 2
    return 3


def _component_duration(q4: Any) -> Optional[int]:
    h = _parse_sleep_hours(str(q4)) if q4 is not None else None
    if h is None:
        return None
    if h > 7:
        return 0
    if h >= 6:
        return 1
    if h >= 5:
        return 2
    return 3


def _component_efficiency(q1: Any, q3: Any, q4: Any) -> Optional[int]:
    bed = _parse_time_minutes(str(q1)) if q1 is not None else None
    wake = _parse_time_minutes(str(q3)) if q3 is not None else None
    hours = _parse_sleep_hours(str(q4)) if q4 is not None else None
    if bed is None or wake is None or hours is None:
        return None
    time_in_bed = wake - bed
    if time_in_bed <= 0:
        time_in_bed += 24 * 60
    if time_in_bed <= 0:
        return None
    eff = 100.0 * (hours * 60.0 / time_in_bed)
    if eff > 85:
        return 0
    if eff >= 75:
        return 1
    if eff >= 65:
        return 2
    return 3


def _component_disturbances(data: Dict[str, Dict]) -> Optional[int]:
    keys = ["5b", "5c", "5d", "5e", "5f", "5g", "5h", "5i", "5j2"]
    partner_keys = ["10a", "10b", "10c", "10d"]
    scores = []
    for k in keys:
        for src in ("psqi_fragebogen_1.yml", "psqi_fragebogen_2.yml"):
            v = data.get(src, {}).get(k)
            s = _freq_score(v)
            if s is not None:
                scores.append(s)
    for k in partner_keys:
        for src in ("psqi_fragebogen_3.yml", "psqi_fragebogen_4.yml"):
            v = data.get(src, {}).get(k)
            s = _freq_score(v)
            if s is not None:
                scores.append(s)
    if not scores:
        return None
    total = sum(scores)
    if total == 0:
        return 0
    if total <= 9:
        return 1
    if total <= 18:
        return 2
    return 3


def score_psqi(sample_id: str, base_path: str = CPS_DATA) -> Optional[Dict[str, float]]:
    files = {
        "psqi_fragebogen_1.yml": _load_yaml(os.path.join(base_path, sample_id, "YAML", "psqi_fragebogen_1.yml")),
        "psqi_fragebogen_2.yml": _load_yaml(os.path.join(base_path, sample_id, "YAML", "psqi_fragebogen_2.yml")),
        "psqi_fragebogen_3.yml": _load_yaml(os.path.join(base_path, sample_id, "YAML", "psqi_fragebogen_3.yml")),
        "psqi_fragebogen_4.yml": _load_yaml(os.path.join(base_path, sample_id, "YAML", "psqi_fragebogen_4.yml")),
    }
    d1, d2, d3 = files["psqi_fragebogen_1.yml"], files["psqi_fragebogen_2.yml"], files["psqi_fragebogen_3.yml"]

    q1 = d1.get("1")
    if q1 is not None and "psqi" in _norm(q1) and ("fehlt" in _norm(q1) or "nicht" in _norm(q1)):
        return None

    comp = {}
    q6 = d2.get("6")
    comp["C1_quality"] = QUALITY_SCORE.get(_norm(q6)) if q6 is not None else None
    comp["C2_latency"] = _component_latency(d1.get("2"), d1.get("5a"))
    comp["C3_duration"] = _component_duration(d1.get("4"))
    comp["C4_efficiency"] = _component_efficiency(d1.get("1"), d1.get("3"), d1.get("4"))
    comp["C5_disturbances"] = _component_disturbances(files)
    comp["C6_medication"] = _freq_score(d3.get("7"))
    q8, q9 = d3.get("8"), d3.get("9")
    s8 = _freq_score(q8)
    s9 = PROBLEM_SCORE.get(_norm(q9)) if q9 is not None else None
    if s8 is not None and s9 is not None:
        t = s8 + s9
        comp["C7_daytime"] = 0 if t == 0 else (1 if t <= 2 else (2 if t <= 4 else 3))
    elif s8 is not None:
        comp["C7_daytime"] = s8
    elif s9 is not None:
        comp["C7_daytime"] = min(s9, 3)
    else:
        comp["C7_daytime"] = None

    n_comp = sum(v is not None for v in comp.values())
    if n_comp < 5:
        return None
    medians = {
        "C1_quality": 1.0, "C2_latency": 1.0, "C3_duration": 1.0,
        "C4_efficiency": 1.0, "C5_disturbances": 1.0, "C6_medication": 0.0, "C7_daytime": 1.0,
    }
    filled = {k: (v if v is not None else medians[k]) for k, v in comp.items()}
    total = float(sum(filled.values()))
    out = {k: float(v) for k, v in comp.items() if v is not None}
    for k, v in filled.items():
        if k not in out:
            out[k] = float(v)
            out[f"{k}_imputed"] = 1.0
    out["PSQI_total"] = total
    out["n_components_scored"] = float(n_comp)
    return out


def _cv_r2(X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    if len(y) < folds + 2:
        return float("nan")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LinearRegression()
    scores = cross_val_score(model, Xs, y, cv=folds, scoring="r2")
    return float(np.mean(scores))


def _fit_r2(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LinearRegression().fit(Xs, y)
    pred = model.predict(Xs)
    r2 = r2_score(y, pred)
    n, p = X.shape
    adj = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    return float(r2), float(adj)


def run_analysis() -> pd.DataFrame:
    metrics = load_valid_metrics()
    rows = []
    for sid in metrics["subject_id"]:
        psqi = score_psqi(sid)
        ahi_info = compute_cps_scorer_ahi(CPS_DATA, sid)
        row = {"subject_id": sid, "ahi_cps_scorer": ahi_info.get("ahi_cps_scorer", np.nan)}
        if psqi:
            row.update(psqi)
        rows.append(row)
    psqi_df = pd.DataFrame(rows)
    df = metrics.merge(psqi_df, on="subject_id", how="left")
    df = df.dropna(subset=["PSQI_total"])

    psqi_cols = [c for c in df.columns if c.startswith("C") or c == "PSQI_total"]
    targets = psqi_cols

    corr_rows = []
    predictors = MANIFOLD_COLS + ["ahi_cps_scorer"]
    for tgt in targets:
        for pred in predictors:
            sub = df[[tgt, pred]].dropna()
            if len(sub) < 8:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[tgt], sub[pred])
            corr_rows.append({"target": tgt, "predictor": pred, "spearman_r": rho, "p_value": p, "n": len(sub)})
    corr_df = pd.DataFrame(corr_rows)

    y = df["PSQI_total"].values
    valid = np.isfinite(y) & np.isfinite(df["ahi_cps_scorer"].values)
    for c in MANIFOLD_COLS:
        valid &= np.isfinite(df[c].values)
    dfv = df.loc[valid].copy()
    y = dfv["PSQI_total"].values

    models = {
        "AHI_only": dfv[["ahi_cps_scorer"]].values,
        "Manifold_only": dfv[MANIFOLD_COLS].values,
        "AHI_plus_Manifold": dfv[["ahi_cps_scorer"] + MANIFOLD_COLS].values,
    }
    reg_rows = []
    for name, X in models.items():
        r2, adj = _fit_r2(X, y)
        cv = _cv_r2(X, y)
        reg_rows.append({"model": name, "R2": r2, "Adj_R2": adj, "CV_R2": cv, "n_features": X.shape[1], "n": len(y)})

    best_rows = []
    for tgt in [c for c in psqi_cols if c != "PSQI_total"]:
        sub_rows = corr_df[(corr_df["target"] == tgt) & corr_df["predictor"].isin(predictors)].copy()
        sub_rows["abs_r"] = sub_rows["spearman_r"].abs()
        if sub_rows.empty or sub_rows["abs_r"].isna().all():
            continue
        best = sub_rows.loc[sub_rows["abs_r"].idxmax()]
        best_rows.append(
            {
                "component": tgt,
                "best_predictor": best["predictor"],
                "spearman_r": best["spearman_r"],
                "p_value": best["p_value"],
            }
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = os.path.join(OUTPUT_DIR, f"psqi_manifold_merged_{ts}.csv")
    corr_path = os.path.join(OUTPUT_DIR, f"psqi_correlations_{ts}.csv")
    reg_path = os.path.join(OUTPUT_DIR, f"psqi_regression_compare_{ts}.csv")
    df.to_csv(merged_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    pd.DataFrame(reg_rows).to_csv(reg_path, index=False)

    summary = {
        "n_subjects": int(len(dfv)),
        "psqi_total_mean": float(dfv["PSQI_total"].mean()),
        "ahi_mean": float(dfv["ahi_cps_scorer"].mean()),
        "regression_comparison": reg_rows,
        "best_predictors_per_component": best_rows,
        "files": {"merged": merged_path, "correlations": corr_path, "regression": reg_path},
    }
    summary_path = os.path.join(OUTPUT_DIR, f"psqi_analysis_summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f": {len(dfv)}")
    print("\n=== PSQI_total  () ===")
    for r in reg_rows:
        print(f"  {r['model']:20s}  R2={r['R2']:.3f}  Adj_R2={r['Adj_R2']:.3f}  CV_R2={r['CV_R2']:.3f}")

    print("\n===  vs AHI： PSQI_total  Spearman  ===")
    for pred in predictors:
        row = corr_df[(corr_df["target"] == "PSQI_total") & (corr_df["predictor"] == pred)].iloc[0]
        sig = "*" if row["p_value"] < 0.05 else ""
        print(f"  {pred:16s}  r={row['spearman_r']:+.3f}  p={row['p_value']:.4f}{sig}")

    print("\n===  PSQI  ===")
    for b in best_rows:
        sig = "*" if b["p_value"] < 0.05 else ""
        print(f"  {b['component']:16s} <- {b['best_predictor']:16s}  r={b['spearman_r']:+.3f}  p={b['p_value']:.4f}{sig}")

    print(f"\n: {summary_path}")
    return df


if __name__ == "__main__":
    run_analysis()
