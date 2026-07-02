"""PSQI component LASSO with manifold, AHI, ARI, and demographics."""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LassoCV
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

if CPS_ROOT not in sys.path:
    sys.path.insert(0, CPS_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_ari_correlation import compute_ahi_flow, compute_ari  # noqa: E402
from analyze_psqi_correlation import score_psqi  # noqa: E402
from manifold_metrics import METRIC_NAMES

MANIFOLD_COLS = [f"{m}_{suffix}" for m in METRIC_NAMES for suffix in ("std", "mean")]
DEMO_COLS = ["age_ord", "bmi_ord"]
CLINICAL_COLS = ["ahi_flow_only", "ARI_all"]
ALL_PREDICTORS = MANIFOLD_COLS + CLINICAL_COLS + DEMO_COLS
TARGETS = ["C1_quality", "C5_disturbances"]

AGE_MAP = {"<50": 0, "50-60": 1, "60-70": 2, ">70": 3}
BMI_MAP = {"<18.5": 0, "18.5-25": 1, "25-30": 2, ">30": 3}


def _load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _norm_bin(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s or s in ("unknown", "nan", "none"):
        return None
    return s


def load_demographics(sample_id: str, base_path: str = CPS_DATA) -> Dict[str, Optional[float]]:
    path = os.path.join(base_path, sample_id, "YAML", "allgemeiner_schlaffragebogen_1.yml")
    data = _load_yaml(path)
    age_raw = _norm_bin(data.get("0"))
    bmi_raw = _norm_bin(data.get("1"))
    return {
        "age_bin": age_raw,
        "bmi_bin": bmi_raw,
        "age_ord": float(AGE_MAP[age_raw]) if age_raw in AGE_MAP else np.nan,
        "bmi_ord": float(BMI_MAP[bmi_raw]) if bmi_raw in BMI_MAP else np.nan,
    }


def load_merged_table() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_CSV)
    metrics = metrics[~metrics["subject_id"].isin(SKIP_SUBJECTS)]
    metrics = metrics[(metrics["n_epochs"] > 0) & metrics["LCI_std"].notna()]

    rows = []
    for sid in metrics["subject_id"]:
        psqi = score_psqi(sid)
        ari = compute_ari(CPS_DATA, sid, eeg_only=False)
        demo = load_demographics(sid)
        row = {
            "subject_id": sid,
            "ahi_flow_only": compute_ahi_flow(CPS_DATA, sid),
            **demo,
        }
        if ari:
            row["ARI_all"] = ari["ARI"]
        if psqi:
            for k in TARGETS + ["PSQI_total"]:
                if k in psqi:
                    row[k] = psqi[k]
        rows.append(row)
    clinical = pd.DataFrame(rows)
    return metrics.merge(clinical, on="subject_id", how="left")


def _fit_lasso(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_folds: int = 5,
) -> Dict:
    n, p = X.shape
    if n < cv_folds + 2:
        cv_folds = max(2, n - 1)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lasso = LassoCV(cv=cv_folds, random_state=0, max_iter=10000, alphas=80)
    lasso.fit(Xs, y)

    selected = [
        {"feature": name, "coef": float(c), "abs_coef": float(abs(c))}
        for name, c in zip(feature_names, lasso.coef_)
        if abs(c) > 1e-8
    ]
    selected.sort(key=lambda d: d["abs_coef"], reverse=True)

    pred = lasso.predict(Xs)
    r2 = float(r2_score(y, pred))
    adj = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    cv_r2 = float(np.mean(cross_val_score(lasso, Xs, y, cv=cv_folds, scoring="r2")))

    return {
        "alpha": float(lasso.alpha_),
        "n_selected": len(selected),
        "selected_features": selected,
        "R2": r2,
        "Adj_R2": float(adj),
        "CV_R2": cv_r2,
        "n": int(n),
        "n_features_in": int(p),
    }


def _spearman_table(df: pd.DataFrame, target: str) -> List[Dict]:
    rows = []
    sub = df.dropna(subset=[target])
    for col in ALL_PREDICTORS:
        s = sub[[target, col]].dropna()
        if len(s) < 8:
            rho, p = np.nan, np.nan
        else:
            rho, p = stats.spearmanr(s[target], s[col])
        rows.append({
            "target": target,
            "predictor": col,
            "spearman_r": float(rho),
            "p_value": float(p),
            "n": len(s),
        })
    return rows


def main():
    df = load_merged_table()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_lasso: List[Dict] = []
    all_spear: List[Dict] = []

    for target in TARGETS:
        sub = df.dropna(subset=[target] + ALL_PREDICTORS)
        if len(sub) < len(ALL_PREDICTORS) + 3:
            print(f" {target}:  (n={len(sub)})")
            continue
        y = sub[target].values.astype(np.float64)
        X = sub[ALL_PREDICTORS].values.astype(np.float64)
        res = _fit_lasso(X, y, ALL_PREDICTORS)
        res["target"] = target
        res["feature_set"] = "Full"
        all_lasso.append(res)
        all_spear.extend(_spearman_table(sub, target))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lasso_path = os.path.join(OUTPUT_DIR, f"lasso_psqi_components_{ts}.json")
    spear_path = os.path.join(OUTPUT_DIR, f"lasso_psqi_components_spearman_{ts}.csv")

    n_demo = int(df[["age_ord", "bmi_ord"]].notna().all(axis=1).sum())
    summary = {
        "n_manifold_valid": int(len(df)),
        "skipped_subjects": sorted(SKIP_SUBJECTS),
        "predictors": ALL_PREDICTORS,
        "note_sex": "",
        "age_bins": AGE_MAP,
        "bmi_bins": BMI_MAP,
        "n_with_demographics": n_demo,
        "n_complete_for_lasso": int(len(df.dropna(subset=TARGETS + ALL_PREDICTORS))) if all_lasso else 0,
        "lasso": all_lasso,
    }
    with open(lasso_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    pd.DataFrame(all_spear).to_csv(spear_path, index=False)

    print(f": {len(df)} | +BMI: {n_demo}")
    print(": CPS ， age_ord + bmi_ord ")
    print(f" ({len(ALL_PREDICTORS)}): {', '.join(ALL_PREDICTORS)}\n")

    for target in TARGETS:
        print(f"=== LASSO @ {target} ===")
        matched = [r for r in all_lasso if r["target"] == target]
        if not matched:
            print("  ()\n")
            continue
        r = matched[0]
        sel = ", ".join(f"{x['feature']}({x['coef']:+.3f})" for x in r["selected_features"]) or "(none)"
        print(
            f"  n={r['n']}  CV_R2={r['CV_R2']:+.3f}  R2={r['R2']:.3f}  "
            f"alpha={r['alpha']:.4f}  selected: {sel}"
        )
        print()

    print("===  Spearman (C1 / C5) ===")
    spear_df = pd.DataFrame(all_spear)
    for target in TARGETS:
        sub = spear_df[spear_df["target"] == target].sort_values("p_value")
        print(f"\n{target}:")
        for _, row in sub.iterrows():
            sig = "*" if row["p_value"] < 0.05 else ""
            print(f"  {row['predictor']:16s} r={row['spearman_r']:+.3f}  p={row['p_value']:.4f}{sig}")

    print(f"\n: {lasso_path}")
    print(f": {spear_path}")


if __name__ == "__main__":
    main()
