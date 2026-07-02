"""DREAMT outcome correlations."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

_DREAMT = os.path.dirname(os.path.abspath(__file__))
_CPS = os.path.join(os.path.dirname(_DREAMT), "cps_pilot")
for p in (_DREAMT, _CPS):
    if p not in sys.path:
        sys.path.insert(0, p)

from analyze_mfi_partial import rank_residual  # noqa: E402
from config_dreamt import OUTCOMES, OUTPUT_DIR, PARTIAL_COVARIATES  # noqa: E402
from physical_dynamics_metrics import ALL_METRIC_NAMES  # noqa: E402

FEATURE_COLS = [f"{m}_{s}" for m in ALL_METRIC_NAMES for s in ("std", "mean")]
PHYSICAL_COLS = [f"{m}_{s}" for m in ("MTV", "MTC", "MTE", "MSV", "MRR") for s in ("std", "mean")]


def _latest_metrics_csv() -> str:
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "metrics_dreamt_*.csv")))
    if not paths:
        raise FileNotFoundError("")
    return paths[-1]


def partial_spearman(y: np.ndarray, x: np.ndarray, C: np.ndarray) -> tuple[float, float]:
    yr = rank_residual(y, C)
    xr = rank_residual(x, C)
    r, p = stats.pearsonr(yr, xr)
    return float(r), float(p)


def _correlate(df: pd.DataFrame, feature: str, target: str) -> Dict:
    covs = list(PARTIAL_COVARIATES)
    cols = [feature, target] + covs
    sub = df[cols].dropna()
    n = len(sub)
    if n < 10:
        return {
            "feature": feature, "target": target, "n": n,
            "raw_r": np.nan, "raw_p": np.nan,
            "partial_r": np.nan, "partial_p": np.nan,
        }
    y = sub[target].values.astype(float)
    x = sub[feature].values.astype(float)
    C = sub[covs].values.astype(float)
    raw_r, raw_p = stats.spearmanr(x, y)
    pr, pp = partial_spearman(y, x, C)
    return {
        "feature": feature,
        "target": target,
        "n": n,
        "raw_r": float(raw_r),
        "raw_p": float(raw_p),
        "partial_r": float(pr),
        "partial_p": float(pp),
    }


def analyze(metrics_csv: str | None = None) -> pd.DataFrame:
    path = metrics_csv or _latest_metrics_csv()
    df = pd.read_csv(path)
    print(f": {path}  n={len(df)}")

    results = []
    for target in OUTCOMES:
        for feat in FEATURE_COLS:
            if feat not in df.columns:
                continue
            results.append(_correlate(df, feat, target))

    res_df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    corr_path = os.path.join(OUTPUT_DIR, f"dreamt_outcome_correlations_{ts}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"dreamt_outcome_correlations_{ts}.json")
    md_path = os.path.join(OUTPUT_DIR, f"")

    res_df.to_csv(corr_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics_csv": path, "results": results}, f, indent=2, ensure_ascii=False)

    lines = [
        f"",
        "",
        f"",
        "",
        "",
        "",
    ]

    for target in OUTCOMES:
        sub = res_df[res_df["target"] == target].copy()
        sub["abs_r"] = sub["raw_r"].abs()
        top = sub.sort_values("abs_r", ascending=False).head(10)
        sig = sub[sub["raw_p"] < 0.05]
        lines += [
            f"## {target}",
            "",
            f"",
            "",
            "",
            "|------|---|-------|-------|-----------|-----------|",
        ]
        print(f"\n=== {target} (Top 5 |r|) ===")
        for _, row in top.head(5).iterrows():
            star = "*" if row["raw_p"] < 0.05 else ("†" if row["raw_p"] < 0.10 else "")
            print(
                f"  {row['feature']:12s} raw r={row['raw_r']:+.3f} p={row['raw_p']:.4f}{star}  "
                f"partial r={row['partial_r']:+.3f} p={row['partial_p']:.4f}"
            )
        for _, row in top.iterrows():
            lines.append(
                f"| {row['feature']} | {int(row['n'])} | {row['raw_r']:+.3f} | {row['raw_p']:.4f} | "
                f"{row['partial_r']:+.3f} | {row['partial_p']:.4f} |"
            )
        lines.append("")

    lines += ["", ""]
    phys_sig = res_df[(res_df["feature"].isin(PHYSICAL_COLS)) & (res_df["raw_p"] < 0.05)]
    if phys_sig.empty:
        lines.append("。")
    else:
        for _, row in phys_sig.sort_values("raw_p").iterrows():
            lines.append(
                f"- **{row['feature']}** ~ {row['target']}: r={row['raw_r']:+.3f}, p={row['raw_p']:.4f}"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n: {corr_path}")
    print(f": {md_path}")
    return res_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, default="")
    args = parser.parse_args()
    analyze(args.metrics_csv or None)


if __name__ == "__main__":
    main()
