"""Exploratory factor analysis of manifold metrics across cohorts."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CPS = os.path.join(_ROOT, "cps_pilot")
_DREAMT = os.path.join(_ROOT, "dreamt_pilot")
_BIDSLAB = os.path.join(_ROOT, "bidslab_pilot")
for p in (_CPS, _DREAMT, _BIDSLAB):
    if p not in sys.path:
        sys.path.insert(0, p)

from analyze_lasso_psqi_components import OUTPUT_DIR as CPS_OUT, SKIP_SUBJECTS  # noqa: E402
from config_dreamt import OUTCOMES as DREAMT_OUTCOMES, OUTPUT_DIR as DREAMT_OUT  # noqa: E402
from config_bidslab import OUTCOMES as BIDSLAB_OUTCOMES, OUTPUT_DIR as BIDSLAB_OUT  # noqa: E402
from manifold_fa import (  # noqa: E402
    cols_for_feature_set,
    compute_mean_std_coupling,
    correlate_factor_scores,
    prepare_feature_table,
    run_manifold_fa,
    save_fa_bundle,
)

COHORT_SPECS: Dict[str, Dict] = {
    "cps": {
        "label": "CPS",
        "out_dir": CPS_OUT,
        "csv_glob": os.path.join(CPS_OUT, "metrics_extended_physical_*.csv"),
        "skip_ids": SKIP_SUBJECTS,
        "id_col": "subject_id",
        "aggregate": None,
        "outcomes": ("ESS_total", "C1_quality"),
    },
    "dreamt_psg": {
        "label": "DREAMT PSG teacher",
        "out_dir": DREAMT_OUT,
        "csv_glob": os.path.join(DREAMT_OUT, "metrics_dreamt_*.csv"),
        "skip_ids": None,
        "id_col": "subject_id",
        "aggregate": None,
        "outcomes": DREAMT_OUTCOMES,
    },
    "dreamt_student": {
        "label": "DREAMT wearable student OOF",
        "out_dir": DREAMT_OUT,
        "csv_glob": os.path.join(DREAMT_OUT, "distill", "manifold_student_pooled_*.csv"),
        "skip_ids": None,
        "id_col": "subject_id",
        "aggregate": None,
        "outcomes": DREAMT_OUTCOMES,
    },
    "bidslab": {
        "label": "Bidslab",
        "out_dir": BIDSLAB_OUT,
        "csv_glob": os.path.join(BIDSLAB_OUT, "metrics_bidslab_fixed_*.csv"),
        "skip_ids": None,
        "id_col": "subject_id",
        "aggregate": "subject_mean",
        "outcomes": BIDSLAB_OUTCOMES,
    },
}


def _latest_csv(glob_pat: str) -> str:
    paths = sorted(glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"")
    return paths[-1]


def _out_dir_for(key: str, spec: Dict) -> str:
    if key == "bidslab":
        return os.path.join(BIDSLAB_OUT, "analysis")
    return spec["out_dir"]


def _feature_set_label(feature_set: str) -> str:
    return {"combined": "", "std": "", "mean": ""}[feature_set]


def run_cohort(
    key: str,
    *,
    csv_path: Optional[str] = None,
    n_factors: int = 3,
    feature_set: str = "combined",
) -> Dict:
    spec = COHORT_SPECS[key]
    path = csv_path or _latest_csv(spec["csv_glob"])
    raw = pd.read_csv(path)
    feat_cols_in = cols_for_feature_set(feature_set)

    feat_df, feat_cols, dropped, n_before = prepare_feature_table(
        raw,
        id_col=spec["id_col"],
        skip_ids=spec["skip_ids"],
        feature_cols=feat_cols_in,
        aggregate=spec["aggregate"],
    )
    if dropped:
        print(f"  [{key}|{feature_set}]  ({len(dropped)}): {', '.join(dropped)}")

    agg_note = ""
    meta = {"feature_set": feature_set}
    if spec["aggregate"] == "subject_mean":
        agg_note = f""
        meta["n_nights_raw"] = n_before
        meta["aggregation"] = "subject_mean"

    res = run_manifold_fa(
        feat_df, feat_cols, id_col=spec["id_col"], n_factors=n_factors
    )

    ocols = [c for c in spec["outcomes"] if c in raw.columns]
    if spec["aggregate"] == "subject_mean":
        outcomes_df = raw.groupby(spec["id_col"], as_index=False)[ocols].mean(numeric_only=True)
    else:
        outcomes_df = raw[[spec["id_col"]] + ocols].drop_duplicates(subset=[spec["id_col"]])

    outcome_corr = correlate_factor_scores(
        res["scores"], outcomes_df, spec["outcomes"], id_col=spec["id_col"]
    )

    cohort_key = f"{key}_{feature_set}" if feature_set != "combined" else key
    label = f"{spec['label']} · {_feature_set_label(feature_set)}"

    paths = save_fa_bundle(
        res,
        cohort_key=cohort_key,
        out_dir=_out_dir_for(key, spec),
        metrics_csv=path,
        cohort_label=label,
        dropped=dropped,
        aggregate_note=agg_note,
        outcome_corr=outcome_corr,
        meta=meta,
    )

    print(f"\n=== {label} ({cohort_key}) ===")
    print(f"  csv: {path}")
    print(
        f"  n={res['n_units']} | p={res['n_features_in']} | "
        f"KMO={res['kmo_overall']:.3f} | Bartlett p={res['bartlett_p']:.2e}"
    )
    for det in res["factor_detail"]:
        sal = ", ".join(det["salient_vars"][:6])
        if len(det["salient_vars"]) > 6:
            sal += ", ..."
        print(f"  {det['factor']} ({det['variance_explained']*100:.1f}%): {sal}")
    if not outcome_corr.empty:
        best = outcome_corr.sort_values("p").iloc[0]
        print(
            f"  top outcome: {best['factor']}~{best['outcome']} "
            f"r={best['spearman_r']:+.3f} p={best['p']:.4f}"
        )
    print(f"  -> {paths['md']}")

    return {
        "key": key,
        "feature_set": feature_set,
        "paths": paths,
        "result": res,
        "outcome_corr": outcome_corr,
    }


def run_coupling_report(key: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    spec = COHORT_SPECS[key]
    path = csv_path or _latest_csv(spec["csv_glob"])
    raw = pd.read_csv(path)
    coup = compute_mean_std_coupling(
        raw,
        id_col=spec["id_col"],
        skip_ids=spec["skip_ids"],
        aggregate=spec["aggregate"],
    )
    coup.insert(0, "cohort", key)
    finite = coup["spearman_r"].dropna()
    mean_abs = float(finite.abs().mean()) if len(finite) else float("nan")
    n_strong = int((finite.abs() >= 0.5).sum()) if len(finite) else 0
    print(
        f"\n  [{key}] mean-std coupling: mean|rho|={mean_abs:.3f}, "
        f"|rho|>=0.5: {n_strong}/{len(finite)} metrics"
    )
    for _, row in coup.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).iterrows():
        if not np.isfinite(row["spearman_r"]):
            continue
        print(f"    {row['metric']:4s} rho={row['spearman_r']:+.3f} p={row['p']:.4f}")
    return coup


def _resolve_feature_sets(mode: str) -> List[str]:
    if mode == "split":
        return ["std", "mean"]
    if mode == "all_split":
        return ["combined", "std", "mean"]
    return [mode]


def main():
    ap = argparse.ArgumentParser(description="CPS / DREAMT / Bidslab  EFA")
    ap.add_argument(
        "--cohort",
        choices=list(COHORT_SPECS) + ["all"],
        default="all",
    )
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--n_factors", type=int, default=3)
    ap.add_argument(
        "--feature-set",
        choices=["combined", "std", "mean", "split", "all_split"],
        default="split",
        help="combined=18; split= std+mean ; all_split=",
    )
    ap.add_argument("--no-coupling", action="store_true", help=" mean↔std ")
    args = ap.parse_args()

    keys = list(COHORT_SPECS) if args.cohort == "all" else [args.cohort]
    feature_sets = _resolve_feature_sets(args.feature_set)
    coupling_rows: List[pd.DataFrame] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for key in keys:
        csv_override = args.csv if len(keys) == 1 and args.csv else None
        if not args.no_coupling:
            coupling_rows.append(run_coupling_report(key, csv_override))
        for fs in feature_sets:
            run_cohort(
                key,
                csv_path=csv_override,
                n_factors=args.n_factors,
                feature_set=fs,
            )

    if coupling_rows:
        coup_all = pd.concat(coupling_rows, ignore_index=True)
        coup_path = os.path.join(_ROOT, "output", f"manifold_mean_std_coupling_{ts}.csv")
        os.makedirs(os.path.dirname(coup_path), exist_ok=True)
        coup_all.to_csv(coup_path, index=False)
        print(f"\n: {coup_path}")


if __name__ == "__main__":
    main()
