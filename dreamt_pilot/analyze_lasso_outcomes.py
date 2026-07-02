"""LASSO outcome models."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

_DREAMT = os.path.dirname(os.path.abspath(__file__))
_CPS = os.path.join(os.path.dirname(_DREAMT), "cps_pilot")
for p in (_DREAMT, _CPS):
    if p not in sys.path:
        sys.path.insert(0, p)

from analyze_dreamt_outcomes import _correlate  # noqa: E402
from analyze_lasso_psqi_components import _fit_lasso  # noqa: F401
from config_dreamt import OUTCOMES, OUTPUT_DIR, PARTIAL_COVARIATES  # noqa: E402
from manifold_lasso import MANIFOLD_COLS, md_lasso_table, run_lasso_target, selected_by_target  # noqa: E402

DATA_SOURCES = {
    "psg_teacher": {
        "label": "",
        "glob": os.path.join(OUTPUT_DIR, "metrics_dreamt_*.csv"),
    },
    "wearable_student": {
        "label": "",
        "glob": os.path.join(OUTPUT_DIR, "distill", "manifold_student_pooled_*.csv"),
    },
}
COVARIATES = list(PARTIAL_COVARIATES)
FEATURE_SETS = {
    "manifold_18+demo": MANIFOLD_COLS + COVARIATES,
    "manifold_18_only": MANIFOLD_COLS,
}


def _latest(glob_pat: str) -> str:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"")
    return paths[-1]


def _run_lasso_grid(df: pd.DataFrame, feature_set: str) -> List[Dict]:
    preds = FEATURE_SETS[feature_set]
    rows = []
    for target in OUTCOMES:
        if target not in df.columns:
            continue
        r = run_lasso_target(df, target, preds)
        r["feature_set"] = feature_set
        rows.append(r)
    return rows


def _followup_correlations(df: pd.DataFrame, selected: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for target, feats in selected.items():
        for feat in feats:
            if feat not in df.columns:
                continue
            rows.append(_correlate(df, feat, target))
    return pd.DataFrame(rows)


def analyze_source(source_key: str, csv_path: str | None = None) -> Dict:
    spec = DATA_SOURCES[source_key]
    path = csv_path or _latest(spec["glob"])
    df = pd.read_csv(path)
    print(f"\n[{source_key}] {spec['label']}")
    print(f"  : {path}  n={len(df)}")

    primary_set = "manifold_18+demo"
    lasso_rows = _run_lasso_grid(df, primary_set)
    selected = selected_by_target(lasso_rows)

    print(f"\n  LASSO @ {primary_set} ():")
    for r in lasso_rows:
        if r.get("status") != "ok":
            print(f"    {r['target']}: skipped n={r.get('n')}")
            continue
        mf = ", ".join(r["selected_manifold"]) or "(none)"
        print(
            f"    {r['target']}: n={r['n']} CV_R2={r['CV_R2']:+.3f} "
            f"-> {mf}"
        )

    corr_df = _followup_correlations(df, selected)
    return {
        "source": source_key,
        "label": spec["label"],
        "csv": path,
        "n": int(len(df)),
        "feature_set": primary_set,
        "lasso": lasso_rows,
        "selected_manifold_by_target": selected,
        "followup_correlations": corr_df.to_dict(orient="records"),
    }


def main():
    ap = argparse.ArgumentParser(description="DREAMT LASSO + ")
    ap.add_argument(
        "--source",
        choices=list(DATA_SOURCES) + ["all"],
        default="all",
        help="psg_teacher / wearable_student / all",
    )
    ap.add_argument("--csv", type=str, default="", help=" metrics CSV")
    args = ap.parse_args()

    keys = list(DATA_SOURCES) if args.source == "all" else [args.source]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_blocks: List[Dict] = []

    for key in keys:
        block = analyze_source(key, args.csv if len(keys) == 1 and args.csv else None)
        all_blocks.append(block)

    out_dir = OUTPUT_DIR
    json_path = os.path.join(out_dir, f"dreamt_lasso_{ts}.json")
    corr_path = os.path.join(out_dir, f"dreamt_lasso_selected_correlations_{ts}.csv")
    md_path = os.path.join(out_dir, f"")

    corr_frames = []
    for b in all_blocks:
        if b["followup_correlations"]:
            cdf = pd.DataFrame(b["followup_correlations"])
            cdf.insert(0, "source", b["source"])
            corr_frames.append(cdf)
    if corr_frames:
        pd.concat(corr_frames, ignore_index=True).to_csv(corr_path, index=False)

    summary = {"timestamp": ts, "covariates": COVARIATES, "cohorts": all_blocks}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        f"",
        "",
        f"",
        "",
        "",
    ]
    for b in all_blocks:
        lines += [f"## {b['label']} (`{b['csv']}`，n={b['n']})", ""]
        lines.extend(md_lasso_table(b["lasso"]))
        lines += ["", "", ""]
        sel = b["selected_manifold_by_target"]
        if not any(sel.values()):
            lines.append("__")
        else:
            lines.append("| source |  |  | n | raw r | raw p | partial r | partial p |")
            lines.append("|--------|------|------|---|-------|-------|-----------|-----------|")
            for row in b["followup_correlations"]:
                lines.append(
                    f"| {b['source']} | {row['target']} | {row['feature']} | {int(row['n'])} | "
                    f"{row['raw_r']:+.3f} | {row['raw_p']:.4f} | "
                    f"{row['partial_r']:+.3f} | {row['partial_p']:.4f} |"
                )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n: {json_path}")
    print(f": {corr_path}")
    print(f": {md_path}")


if __name__ == "__main__":
    main()
