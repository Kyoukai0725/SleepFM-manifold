"""Cross-cohort super-metrics: Activity, Stability, Irregularity."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CPS = os.path.join(_ROOT, "cps_pilot")
_DREAMT = os.path.join(_ROOT, "dreamt_pilot")
_BIDSLAB = os.path.join(_ROOT, "bidslab_pilot")
_APNEA = os.path.join(_ROOT, "apnea_pilot")
for p in (_CPS, _DREAMT, _BIDSLAB, _APNEA):
    if p not in sys.path:
        sys.path.insert(0, p)

from analyze_ari_correlation import CPS_DATA, _sleep_hours, compute_ahi_flow, compute_ari  # noqa: E402
from analyze_lasso_psqi_components import OUTPUT_DIR as CPS_OUT, SKIP_SUBJECTS  # noqa: E402
from config_dreamt import (  # noqa: E402
    DISTILL_OUTPUT_DIR as DREAMT_DISTILL_OUT,
    OUTPUT_DIR as DREAMT_OUT,
    TEACHER_EMB_CACHE,
)
from config_bidslab import NIGHT_CACHE, OUTPUT_DIR as BIDSLAB_OUT  # noqa: E402
from config_apnea import OUTPUT_DIR as APNEA_OUT  # noqa: E402
from physical_dynamics_metrics import analyze_irregularity_metrics  # noqa: E402

SUPER_METRICS: Dict[str, List[str]] = {
    "Trajectory_Activity": ["LCI", "MTV", "MTC"],
    "Stability": ["SMI", "VTI"],
    "Trajectory_Irregularity": ["MFI"],
}

SUPER_METRICS_MFI_LZC: Dict[str, List[str]] = {
    **{k: v[:] for k, v in SUPER_METRICS.items() if k != "Trajectory_Irregularity"},
    "Trajectory_Irregularity": ["MFI", "LZC"],
}

TARGET_OUTCOMES = {
    "ESS": "ESS_total",
    "C1": "C1_quality",
    "ARI": "ARI",
    "AHI": "AHI",
    "TST": "TST_hours",
}

OUTCOME_COLUMN_ALIASES = {
    "AHI": ("AHI", "ahi_flow_only"),
    "ARI": ("ARI", "ARI_all"),
}

HYPOTHESIS = {
    "ESS": {"Trajectory_Activity": "weak", "Stability": "weak", "Trajectory_Irregularity": "strong"},
    "ARI": {"Trajectory_Activity": "strong", "Stability": "weak", "Trajectory_Irregularity": "moderate"},
    "TST": {"Trajectory_Activity": "moderate", "Stability": "strong", "Trajectory_Irregularity": "weak"},
    "AHI": {"Trajectory_Activity": "moderate", "Stability": "weak", "Trajectory_Irregularity": "weak"},
    "C1": {"Trajectory_Activity": "weak", "Stability": "weak", "Trajectory_Irregularity": "strong"},
}

COHORT_SPECS = {
    "cps": {
        "label": "CPS",
        "csv_glob": os.path.join(CPS_OUT, "metrics_extended_physical_*.csv"),
        "skip_ids": SKIP_SUBJECTS,
        "aggregate": None,
        "enrich": "cps_clinical",
        "outcomes_available": ("ESS", "C1", "ARI", "AHI", "TST"),
    },
    "dreamt_psg": {
        "label": "DREAMT PSG",
        "csv_glob": os.path.join(DREAMT_OUT, "metrics_dreamt_*.csv"),
        "skip_ids": None,
        "aggregate": None,
        "enrich": None,
        "outcomes_available": ("ARI", "AHI", "TST"),
    },
    "dreamt_student": {
        "label": "DREAMT wearable OOF",
        "csv_glob": os.path.join(DREAMT_OUT, "distill", "manifold_student_pooled_*.csv"),
        "skip_ids": None,
        "aggregate": None,
        "enrich": None,
        "outcomes_available": ("ARI", "AHI", "TST"),
    },
    "bidslab": {
        "label": "Bidslab",
        "csv_glob": os.path.join(BIDSLAB_OUT, "metrics_bidslab_fixed_*.csv"),
        "skip_ids": None,
        "aggregate": "subject_mean",
        "enrich": None,
        "outcomes_available": ("TST",),
    },
    "apnea": {
        "label": "APNEA",
        "csv_glob": os.path.join(APNEA_OUT, "metrics_apnea_*.csv"),
        "skip_ids": None,
        "aggregate": None,
        "enrich": None,
        "outcomes_available": ("ARI", "AHI"),
    },
}


def _latest(glob_pat: str) -> str:
    paths = sorted(glob(glob_pat))
    if not paths:
        raise FileNotFoundError(glob_pat)
    return paths[-1]


def _resolve_outcome_column(df: pd.DataFrame, outcome_key: str) -> Optional[str]:
    """Helper."""
    if outcome_key in OUTCOME_COLUMN_ALIASES:
        for col in OUTCOME_COLUMN_ALIASES[outcome_key]:
            if col in df.columns:
                return col
        return None
    col = TARGET_OUTCOMES.get(outcome_key)
    return col if col in df.columns else None


def _enrich_cps_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Helper."""
    rows = []
    for sid in df["subject_id"]:
        row: Dict[str, float | str] = {"subject_id": sid}
        row["ahi_flow_only"] = compute_ahi_flow(CPS_DATA, sid)
        row["AHI"] = row["ahi_flow_only"]
        ari = compute_ari(CPS_DATA, sid, eeg_only=False)
        if ari:
            row["ARI"] = ari["ARI"]
        try:
            row["TST_hours"] = _sleep_hours(CPS_DATA, sid)
        except Exception:
            row["TST_hours"] = np.nan
        rows.append(row)
    clin = pd.DataFrame(rows)
    drop = [c for c in ("ARI", "AHI", "ahi_flow_only", "TST_hours") if c in df.columns]
    out = df.drop(columns=drop, errors="ignore").merge(clin, on="subject_id", how="left")
    return out


def _load_cohort_df(spec: Dict, *, aggregate: bool = True) -> pd.DataFrame:
    df = pd.read_csv(_latest(spec["csv_glob"]))
    if spec["skip_ids"]:
        df = df[~df["subject_id"].isin(spec["skip_ids"])]
    if spec.get("enrich") == "cps_clinical":
        df = _enrich_cps_clinical(df)
    if aggregate and spec["aggregate"] == "subject_mean":
        num = [c for c in df.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(df[c])]
        df = df.groupby("subject_id", as_index=False)[num].mean(numeric_only=True)
    return df


def _needs_lzc_backfill(df: pd.DataFrame) -> bool:
    return not {"LZC_mean", "LZC_std"}.issubset(df.columns)


def _load_pkl_embeddings(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict):
        return {str(k): np.asarray(v) for k, v in raw.items()}
    return {}


def _load_bidslab_night_embeddings() -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    cache = Path(NIGHT_CACHE)
    if not cache.is_dir():
        return out
    for npz in cache.glob("*_night*.npz"):
        z = np.load(npz, allow_pickle=True)
        if "embedding" not in z:
            continue
        stem = npz.stem
        if "_night" not in stem:
            continue
        sid, night_s = stem.rsplit("_night", 1)
        out[f"{sid}__night{int(night_s)}"] = np.asarray(z["embedding"])
    return out


def _load_dreamt_student_embeddings() -> Dict[str, np.ndarray]:
    """Helper."""
    try:
        import torch
    except ImportError:
        print("  [dreamt_student]  torch，", flush=True)
        return {}

    from train_wearable_distill import (  # noqa: WPS433
        _extract_student_embeddings,
        _load_teacher,
        _make_folds,
    )
    from preprocess_dreamt import list_psg_subjects  # noqa: WPS433
    from wearable_student import WearableSleepFM  # noqa: WPS433

    teacher = _load_teacher()
    sids = [s for s in list_psg_subjects() if s in teacher]
    folds = _make_folds(sids)
    device = torch.device("cpu")
    out: Dict[str, np.ndarray] = {}
    for fid, fold in folds.items():
        ckpt = os.path.join(DREAMT_DISTILL_OUT, f"manifold_student_fold{fid}", "best.pt")
        if not os.path.isfile(ckpt):
            continue
        ckpt_obj = torch.load(ckpt, map_location=device)
        model = WearableSleepFM().to(device)
        model.load_state_dict(ckpt_obj["state_dict"])
        for sid, emb in _extract_student_embeddings(
            model, fold["test"], teacher, device, normalize=False
        ).items():
            out[sid] = emb
    return out


def _augment_lzc(cohort_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Helper."""
    if not _needs_lzc_backfill(df):
        return df

    lzc_cols = ("LZC_mean", "LZC_std")
    emb_map: Dict[str, np.ndarray] = {}
    if cohort_key == "cps":
        emb_map = _load_pkl_embeddings(os.path.join(CPS_OUT, "cache", "subject_embeddings_128d.pkl"))
    elif cohort_key in ("dreamt_psg",):
        emb_map = _load_pkl_embeddings(TEACHER_EMB_CACHE)
    elif cohort_key == "dreamt_student":
        emb_map = _load_dreamt_student_embeddings()
    elif cohort_key == "apnea":
        emb_map = _load_pkl_embeddings(os.path.join(APNEA_OUT, "cache", "embeddings_128d.pkl"))
    elif cohort_key == "bidslab":
        if "night" not in df.columns:
            print(f"  [{cohort_key}]  night ， LZC ", flush=True)
            return df
        emb_map = _load_bidslab_night_embeddings()
        rows = []
        for _, row in df.iterrows():
            sid = str(row["subject_id"])
            night = int(row["night"])
            key = f"{sid}__night{night}"
            emb = emb_map.get(key)
            if emb is None:
                continue
            m = analyze_irregularity_metrics(emb)
            rows.append({"subject_id": sid, "night": night, **{k: m[k] for k in lzc_cols if k in m}})
        if not rows:
            return df
        aug = pd.DataFrame(rows)
        merge_keys = ["subject_id", "night"]
        drop = [c for c in lzc_cols if c in df.columns]
        out = df.drop(columns=drop, errors="ignore").merge(aug, on=merge_keys, how="left")
        print(f"  [{cohort_key}] LZC  n={len(rows)} ", flush=True)
        return out

    if not emb_map:
        print(f"  [{cohort_key}] ， LZC ", flush=True)
        return df

    rows = []
    for sid in df["subject_id"].astype(str):
        emb = emb_map.get(sid)
        if emb is None:
            continue
        m = analyze_irregularity_metrics(emb)
        rows.append({"subject_id": sid, **{k: m[k] for k in lzc_cols if k in m}})
    if not rows:
        return df
    aug = pd.DataFrame(rows)
    drop = [c for c in lzc_cols if c in df.columns]
    out = df.drop(columns=drop, errors="ignore").merge(aug, on="subject_id", how="left")
    print(f"  [{cohort_key}] LZC  n={len(rows)}", flush=True)
    return out


def _partial_covs(cohort: str, outcome_key: str, df: pd.DataFrame) -> List[str]:
    if cohort in ("apnea", "bidslab"):
        return []
    if cohort != "cps":
        if outcome_key == "AHI":
            base = ["AGE", "BMI", "GENDER_M"]
        else:
            base = ["AGE", "BMI", "GENDER_M"]
        return [c for c in base if c in df.columns]
    if outcome_key == "AHI":
        base = ["age_ord", "bmi_ord"]
    else:
        base = ["age_ord", "bmi_ord", "ahi_flow_only"]
    return [c for c in base if c in df.columns]


def _zseries(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=1)
    if sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def _composite(df: pd.DataFrame, bases: Sequence[str], suffix: str) -> pd.Series:
    cols = [f"{b}_{suffix}" for b in bases if f"{b}_{suffix}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    z = pd.DataFrame({c: _zseries(df[c]) for c in cols})
    return z.mean(axis=1)


def _spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < 8:
        return float("nan"), float("nan"), len(sub)
    r, p = stats.spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
    return float(r), float(p), len(sub)


def _partial_spearman(y: np.ndarray, x: np.ndarray, C: np.ndarray) -> Tuple[float, float]:
    _lib = os.path.join(_ROOT, "lib")
    if _lib not in sys.path:
        sys.path.insert(0, _lib)
    from stats_utils import partial_spearman  # noqa: WPS433

    return partial_spearman(y, x, C)


def _corr_cohesion(z_df: pd.DataFrame) -> Tuple[float, float]:
    """Helper."""
    if z_df.shape[1] < 2 or len(z_df) < 8:
        return float("nan"), float("nan")
    if z_df.std(ddof=1).min() < 1e-12:
        return float("nan"), float("nan")
    corr = z_df.corr(method="pearson").values.astype(float)
    p = corr.shape[0]
    off = corr[~np.eye(p, dtype=bool)]
    mean_abs_r = float(np.nanmean(np.abs(off))) if off.size else float("nan")
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.sort(eigvals)[::-1]
    total = float(np.sum(np.clip(eigvals, 0, None)))
    evr1 = float(eigvals[0] / total) if total > 1e-12 else float("nan")
    return mean_abs_r, evr1


def _suffix_covariance_cohesion(
    df: pd.DataFrame,
    bases: Sequence[str],
    *,
    all_bases: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Helper."""
    scores: Dict[str, Tuple[float, float]] = {}
    for suffix in ("mean", "std"):
        cols = [f"{b}_{suffix}" for b in bases if f"{b}_{suffix}" in df.columns]
        sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(cols) >= 2:
            z = pd.DataFrame({c: _zseries(sub[c]) for c in cols})
            scores[suffix] = _corr_cohesion(z)
        elif len(cols) == 1:
            anchor = cols[0]
            peers = [
                f"{b}_{suffix}"
                for b in all_bases
                if b not in bases and f"{b}_{suffix}" in df.columns
            ]
            if not peers:
                scores[suffix] = (float("nan"), float("nan"))
                continue
            anchor_z = _zseries(sub[anchor])
            rs = []
            for pc in peers:
                pair = pd.concat([anchor_z, _zseries(df[pc])], axis=1).dropna()
                if len(pair) < 8 or pair.iloc[:, 1].std(ddof=1) < 1e-12:
                    continue
                r, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                if np.isfinite(r):
                    rs.append(abs(float(r)))
            cohesion = float(np.mean(rs)) if rs else float("nan")
            scores[suffix] = (cohesion, cohesion)
        else:
            scores[suffix] = (float("nan"), float("nan"))

    def _rank(suffix: str) -> Tuple[float, float, int]:
        m, e = scores[suffix]
        m = m if np.isfinite(m) else -1.0
        e = e if np.isfinite(e) else -1.0
        valid = int(np.isfinite(scores[suffix][0]))
        return (m, e, valid)

    best = max(("mean", "std"), key=_rank)
    meta = {
        "mean_cohesion": scores["mean"][0],
        "mean_evr1": scores["mean"][1],
        "std_cohesion": scores["std"][0],
        "std_evr1": scores["std"][1],
        "rule": "multi_component_cov" if len(bases) >= 2 else "singleton_peer_cov",
    }
    return best, meta


ALL_BASES = sorted({b for bs in SUPER_METRICS.values() for b in bs})


def _choose_suffixes(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    suffix_choice: Dict[str, str] = {}
    suffix_meta: Dict[str, Dict[str, float]] = {}
    for sname, bases in SUPER_METRICS.items():
        suffix, meta = _suffix_covariance_cohesion(df, bases, all_bases=ALL_BASES)
        suffix_choice[sname] = suffix
        suffix_meta[sname] = meta
    return suffix_choice, suffix_meta


def _strength_label(r: float, p: float) -> str:
    if not np.isfinite(r) or not np.isfinite(p):
        return "—"
    if p < 0.05 and abs(r) >= 0.20:
        return "strong"
    if p < 0.10 or abs(r) >= 0.25:
        return "moderate"
    if abs(r) >= 0.15:
        return "moderate"
    return "weak"


def _match_hypothesis(outcome: str, super_name: str, label: str) -> str:
    exp = HYPOTHESIS.get(outcome, {}).get(super_name, "")
    if not exp or label == "—":
        return ""
    return "Y" if label == exp else "N"


def run_cohort(key: str, *, skip_lzc_backfill: bool = False) -> Dict:
    spec = COHORT_SPECS[key]
    df = _load_cohort_df(spec, aggregate=False)
    if not skip_lzc_backfill:
        df = _augment_lzc(key, df)
    if spec["aggregate"] == "subject_mean":
        num = [c for c in df.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(df[c])]
        df = df.groupby("subject_id", as_index=False)[num].mean(numeric_only=True)

    outcome_cols: Dict[str, str] = {}
    for ok in spec["outcomes_available"]:
        col = _resolve_outcome_column(df, ok)
        if col:
            outcome_cols[ok] = col

    suffix_choice, suffix_meta = _choose_suffixes(df)
    legacy_irreg_suffix, legacy_irreg_meta = _suffix_covariance_cohesion(
        df, ["MFI"], all_bases=ALL_BASES
    )
    for sname, bases in SUPER_METRICS.items():
        df[sname] = _composite(df, bases, suffix_choice[sname])

    rows = []
    for oname, ocol in outcome_cols.items():
        covs = _partial_covs(key, oname, df)
        for sname in SUPER_METRICS:
            raw_r, raw_p, n = _spearman(df[sname], df[ocol])
            pr, pp = float("nan"), float("nan")
            if covs:
                sub = df[[sname, ocol] + covs].dropna()
                if len(sub) >= len(covs) + 8:
                    pr, pp = _partial_spearman(
                        sub[ocol].values.astype(float),
                        sub[sname].values.astype(float),
                        sub[covs].values.astype(float),
                    )
            sl = _strength_label(raw_r, raw_p)
            rows.append(
                {
                    "cohort": key,
                    "cohort_label": spec["label"],
                    "outcome": oname,
                    "outcome_col": ocol,
                    "super_metric": sname,
                    "suffix_used": suffix_choice[sname],
                    "components": "+".join(f"{b}_{suffix_choice[sname]}" for b in SUPER_METRICS[sname]),
                    "n": n,
                    "raw_r": raw_r,
                    "raw_p": raw_p,
                    "partial_r": pr,
                    "partial_p": pp,
                    "strength": sl,
                    "hypothesis": HYPOTHESIS.get(oname, {}).get(sname, ""),
                    "match": _match_hypothesis(oname, sname, sl),
                }
            )

    return {
        "cohort": key,
        "label": spec["label"],
        "n_units": len(df),
        "outcome_cols": outcome_cols,
        "suffix_choice": suffix_choice,
        "suffix_meta": suffix_meta,
        "legacy_irreg": {
            "suffix": legacy_irreg_suffix,
            "meta": legacy_irreg_meta,
        },
        "correlations": rows,
    }


def _write_md(results: List[Dict], path: str) -> None:
    irr_bases = SUPER_METRICS["Trajectory_Irregularity"]
    if irr_bases == ["MFI"]:
        irreg_def = [
            "",
        ]
        cmp_header = ""
    else:
        irreg_def = [
            f"- **Trajectory_Irregularity** = mean(z): **{' + '.join(irr_bases)}**",
            "",
        ]
        cmp_header = ""

    lines = [
        f"",
        "",
        "",
        "",
        "- **Trajectory_Activity** = mean(z): LCI + MTV + MTC",
        "- **Stability** = mean(z): SMI + VTI",
        *irreg_def,
        "",
        "",
        cmp_header,
        "",
    ]
    if len(irr_bases) > 1:
        lines += [
            "",
            "|------|------------------------|------------------------|--------|",
        ]
    else:
        lines += [
            "",
            "|------|---------------------|------|",
        ]
    for res in results:
        leg = res.get("legacy_irreg", {}).get("meta", {})
        ir = res.get("suffix_meta", {}).get("Trajectory_Irregularity", {})
        sc = res["suffix_choice"]
        if len(irr_bases) > 1:
            lines.append(
                f"| {res['label']} | "
                f"{leg.get('mean_cohesion', float('nan')):.2f}/{leg.get('std_cohesion', float('nan')):.2f} | "
                f"{ir.get('mean_cohesion', float('nan')):.2f}/{ir.get('std_cohesion', float('nan')):.2f} | "
                f"{sc['Trajectory_Irregularity']} |"
            )
        else:
            lines.append(
                f"| {res['label']} | "
                f"{ir.get('mean_cohesion', float('nan')):.2f}/{ir.get('std_cohesion', float('nan')):.2f} | "
                f"{sc['Trajectory_Irregularity']} |"
            )

    lines += [
        "",
        "",
        "",
        "",
        "|------|----------|-----------|--------------|-------------------|----------|-----------|",
    ]
    for res in results:
        sc = res["suffix_choice"]
        sm = res.get("suffix_meta", {})
        act = sm.get("Trajectory_Activity", {})
        stab = sm.get("Stability", {})
        ir = sm.get("Trajectory_Irregularity", {})
        lines.append(
            f"| {res['label']} | {sc['Trajectory_Activity']} | {sc['Stability']} | "
            f"{sc['Trajectory_Irregularity']} | "
            f"{act.get('mean_cohesion', float('nan')):.2f}/{act.get('std_cohesion', float('nan')):.2f} | "
            f"{stab.get('mean_cohesion', float('nan')):.2f}/{stab.get('std_cohesion', float('nan')):.2f} | "
            f"{ir.get('mean_cohesion', float('nan')):.2f}/{ir.get('std_cohesion', float('nan')):.2f} |"
        )

    lines += [
        "",
        "",
        "",
        "",
        "",
        "",
        "|------|----------|-----------|--------------|",
    ]
    for res in results:
        sc = res["suffix_choice"]
        lines.append(
            f"| {res['label']} | {sc['Trajectory_Activity']} | {sc['Stability']} | {sc['Trajectory_Irregularity']} |"
        )

    all_outcomes = []
    for res in results:
        for ok in res.get("outcome_cols", {}):
            if ok not in all_outcomes:
                all_outcomes.append(ok)

    lines += ["", "", ""]
    for oname in all_outcomes:
        lines.append(f"### {oname}")
        lines.append("")
        lines.append(
            f""
        )
        lines.append("|------|----------|-----------|--------------|")
        for res in results:
            if oname not in res.get("outcome_cols", {}):
                continue
            sub = [x for x in res["correlations"] if x["outcome"] == oname]
            cells = {x["super_metric"]: f"{x['raw_r']:+.2f} ({x['strength']})" for x in sub}
            lines.append(
                f"| {res['label']} | "
                f"{cells.get('Trajectory_Activity', '—')} | "
                f"{cells.get('Stability', '—')} | "
                f"{cells.get('Trajectory_Irregularity', '—')} |"
            )
        lines.append("")

    if any(r["cohort"] == "cps" for r in results):
        cps = next(r for r in results if r["cohort"] == "cps")
        lines += [
            "",
            "",
            f"",
            "",
        ]

    lines += ["", ""]
    lines.append(
        ""
    )
    lines.append("|--------|---------|-------|--------|---|-------|-------|-----------|-----------|------|")
    for res in results:
        for row in res["correlations"]:
            pr, pp = row["partial_r"], row["partial_p"]
            lines.append(
                f"| {row['cohort_label']} | {row['outcome']} | {row['super_metric']} | {row['suffix_used']} | "
                f"{row['n']} | {row['raw_r']:+.3f} | {row['raw_p']:.4f} | "
                f"{(pr if np.isfinite(pr) else float('nan')):+.3f} | "
                f"{(pp if np.isfinite(pp) else float('nan')):.4f} | {row['strength']} |"
            )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    global SUPER_METRICS, ALL_BASES

    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--cohort", choices=list(COHORT_SPECS) + ["all"], default="all")
    ap.add_argument(
        "--skip-lzc-backfill",
        action="store_true",
        help=" LZC（ LZC  Irregularity  MFI-only）",
    )
    ap.add_argument(
        "--with-lzc",
        action="store_true",
        help="Irregularity  MFI+LZC （ MFI）",
    )
    args = ap.parse_args()

    if args.with_lzc:
        SUPER_METRICS = {k: v[:] for k, v in SUPER_METRICS_MFI_LZC.items()}
    ALL_BASES = sorted({b for bs in SUPER_METRICS.values() for b in bs})

    keys = list(COHORT_SPECS) if args.cohort == "all" else [args.cohort]
    results = [run_cohort(k, skip_lzc_backfill=args.skip_lzc_backfill) for k in keys]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"super_metrics_correlations_{ts}.csv")
    md_path = os.path.join(out_dir, f"super_metrics_cross_cohort_{ts}.md")

    all_rows = []
    for res in results:
        all_rows.extend(res["correlations"])
        sc = res["suffix_choice"]
        print(f"\n=== {res['label']} (n={res['n_units']}) ===")
        print(f"  outcomes: {res['outcome_cols']}")
        print(
            f"  suffix: Act={sc['Trajectory_Activity']} "
            f"Stab={sc['Stability']} Irreg={sc['Trajectory_Irregularity']} "
            f"(covariance rule)"
        )
        sm = res.get("suffix_meta", {})
        for sname in SUPER_METRICS:
            m = sm.get(sname, {})
            print(
                f"    {sname}: mean_coh={m.get('mean_cohesion', float('nan')):.3f} "
                f"std_coh={m.get('std_cohesion', float('nan')):.3f} -> {sc[sname]}"
            )
        for row in sorted(res["correlations"], key=lambda x: (x["outcome"], x["raw_p"])):
            print(
                f"  {row['outcome']:3s} ~ {row['super_metric']:25s} "
                f"r={row['raw_r']:+.3f} p={row['raw_p']:.4f} [{row['strength']}]"
            )

    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    _write_md(results, md_path)
    print(f"\nSaved: {md_path}")
    print(f": {csv_path}")


if __name__ == "__main__":
    main()
