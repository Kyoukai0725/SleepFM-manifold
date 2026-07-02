"""LASSO outcome models."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

_B = Path(__file__).resolve().parent
_CPS = _B.parent / "cps_pilot"
for p in (_B, _CPS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_bidslab_multilevel import run_between_subject, run_pooled_spearman  # noqa: E402
from config_bidslab import OUTCOMES, OUTPUT_DIR  # noqa: E402
from manifold_lasso import MANIFOLD_COLS, md_lasso_table, run_lasso_target, selected_by_target  # noqa: E402

DEFAULT_CSV = Path(OUTPUT_DIR) / "metrics_bidslab_fixed_20260615_131155.csv"
ANALYSIS_DIR = Path(OUTPUT_DIR) / "analysis"


def _latest_csv() -> Path:
    cands = sorted(Path(OUTPUT_DIR).glob("metrics_bidslab_fixed_*.csv"))
    return cands[-1] if cands else DEFAULT_CSV


def _subject_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["subject_id"] + MANIFOLD_COLS + list(OUTCOMES)
    cols = [c for c in cols if c in df.columns]
    return df.groupby("subject_id", as_index=False)[cols].mean(numeric_only=True)


def run_lmm(df: pd.DataFrame, outcome: str, feature: str) -> dict:
    sub = df[[outcome, feature, "subject_id"]].dropna()
    if len(sub) < 20 or sub["subject_id"].nunique() < 5:
        return {"outcome": outcome, "feature": feature, "ok": False, "reason": "insufficient data"}
    formula = f"{outcome} ~ {feature}"
    try:
        model = smf.mixedlm(formula, data=sub, groups=sub["subject_id"])
        result = model.fit(method="lbfgs", maxiter=200, disp=False)
        ci = result.conf_int().loc[feature]
        return {
            "outcome": outcome,
            "feature": feature,
            "ok": True,
            "n_nights": int(len(sub)),
            "n_subjects": int(sub["subject_id"].nunique()),
            "coef": float(result.fe_params[feature]),
            "se": float(result.bse[feature]),
            "pvalue": float(result.pvalues[feature]),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
        }
    except Exception as exc:
        return {"outcome": outcome, "feature": feature, "ok": False, "reason": str(exc)}


def _followup_multilevel(night_df: pd.DataFrame, selected: dict[str, list[str]]) -> tuple[list, list, list]:
    lmm_rows, between_rows, pooled_rows = [], [], []
    for outcome, feats in selected.items():
        for feat in feats:
            if feat not in night_df.columns:
                continue
            lmm_rows.append(run_lmm(night_df, outcome, feat))
            between_rows.append(run_between_subject(night_df, outcome, feat))
            pooled_rows.append(run_pooled_spearman(night_df, outcome, feat))
    return lmm_rows, between_rows, pooled_rows


def main():
    ap = argparse.ArgumentParser(description="Bidslab LASSO +  LMM/")
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else _latest_csv()
    night_df = pd.read_csv(csv_path)
    subj_df = _subject_means(night_df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f": {csv_path}")
    print(f"   n={len(night_df)} |  n={subj_df['subject_id'].nunique()}")

    lasso_rows = []
    for target in OUTCOMES:
        if target not in subj_df.columns:
            continue
        r = run_lasso_target(subj_df, target, MANIFOLD_COLS)
        r["feature_set"] = "manifold_18_only"
        lasso_rows.append(r)

    selected = selected_by_target(lasso_rows)
    print("\nLASSO @  (18 ):")
    for r in lasso_rows:
        if r.get("status") != "ok":
            print(f"  {r['target']}: skipped")
            continue
        mf = ", ".join(r["selected_manifold"]) or "(none)"
        print(f"  {r['target']}: n={r['n']} CV_R2={r['CV_R2']:+.3f} -> {mf}")

    lmm_rows, between_rows, pooled_rows = _followup_multilevel(night_df, selected)

    json_path = ANALYSIS_DIR / f"bidslab_lasso_{ts}.json"
    md_path = ANALYSIS_DIR / f""

    report = {
        "timestamp": ts,
        "csv": str(csv_path),
        "n_nights": int(len(night_df)),
        "n_subjects_lasso": int(len(subj_df)),
        "lasso_level": "subject_means",
        "predictors": MANIFOLD_COLS,
        "lasso": lasso_rows,
        "selected_manifold_by_target": selected,
        "lmm_selected": lmm_rows,
        "between_subject_selected": between_rows,
        "pooled_spearman_selected": pooled_rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        f"",
        "",
        f"",
        "",
        "",
        "",
        "",
        "",
        *md_lasso_table(lasso_rows),
        "",
    ]

    if not any(selected.values()):
        lines.append("_；。_")
    else:
        lines += [
            "",
            "",
            "",
            "|------|------|-----|-----|------|---|",
        ]
        for row in lmm_rows:
            if not row.get("ok"):
                lines.append(f"| {row['outcome']} | {row['feature']} | — | — | — | {row.get('reason')} |")
            else:
                lines.append(
                    f"| {row['outcome']} | {row['feature']} | {row['n_nights']} | {row['n_subjects']} | "
                    f"{row['coef']:+.4f} | {row['pvalue']:.2e} |"
                )
        lines += [
            "",
            "",
            "",
            "",
            "|------|------|-----|---|---|",
        ]
        for row in between_rows:
            if not row.get("ok"):
                lines.append(f"| {row['outcome']} | {row['feature']} | — | — | {row.get('reason')} |")
            else:
                lines.append(
                    f"| {row['outcome']} | {row['feature']} | {row['n_subjects']} | "
                    f"{row['spearman_r']:+.3f} | {row['pvalue']:.4f} |"
                )
        lines += [
            "",
            "",
            "",
            "",
            "|------|------|-----|---|---|",
        ]
        for row in pooled_rows:
            lines.append(
                f"| {row['outcome']} | {row['feature']} | {row['n_nights']} | "
                f"{row['spearman_r']:+.3f} | {row['pvalue']:.2e} |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n: {json_path}")
    print(f": {md_path}")


if __name__ == "__main__":
    main()
