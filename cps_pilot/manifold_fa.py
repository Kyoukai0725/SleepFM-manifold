"""Exploratory factor analysis helpers."""
from __future__ import annotations

import json
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.utils.validation as _sk_validation
from scipy import stats
from sklearn.preprocessing import StandardScaler

_check_array_orig = _sk_validation.check_array


def _check_array_patched(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _check_array_orig(*args, **kwargs)


_sk_validation.check_array = _check_array_patched

import factor_analyzer.factor_analyzer as _fa_mod  # noqa: E402
from factor_analyzer import FactorAnalyzer  # noqa: E402
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo  # noqa: E402

_fa_mod.check_array = _check_array_patched

from manifold_lasso import MANIFOLD_COLS  # noqa: E402
from physical_dynamics_metrics import ALL_METRIC_NAMES  # noqa: E402

MANIFOLD_STD_COLS = [f"{m}_std" for m in ALL_METRIC_NAMES]
MANIFOLD_MEAN_COLS = [f"{m}_mean" for m in ALL_METRIC_NAMES]
FEATURE_SET_MAP = {
    "combined": MANIFOLD_COLS,
    "std": MANIFOLD_STD_COLS,
    "mean": MANIFOLD_MEAN_COLS,
}

LOADING_THRESHOLD = 0.40


def _salient_vars(loadings: pd.DataFrame, fac: str) -> List[str]:
    s = loadings[fac].sort_values(key=lambda x: x.abs(), ascending=False)
    hit = s[s.abs() >= LOADING_THRESHOLD]
    return hit.index.tolist() if len(hit) else [s.index[0]]


def cols_for_feature_set(feature_set: str) -> List[str]:
    if feature_set not in FEATURE_SET_MAP:
        raise ValueError(f"")
    return list(FEATURE_SET_MAP[feature_set])


def compute_mean_std_coupling(
    df: pd.DataFrame,
    *,
    id_col: str = "subject_id",
    skip_ids: Optional[Sequence[str]] = None,
    aggregate: Optional[str] = None,
) -> pd.DataFrame:
    """Helper."""
    work = df.copy()
    if skip_ids:
        work = work[~work[id_col].isin(skip_ids)]
    if aggregate == "subject_mean":
        num = [c for c in MANIFOLD_COLS if c in work.columns]
        work = work.groupby(id_col, as_index=False)[num].mean(numeric_only=True)

    rows = []
    for m in ALL_METRIC_NAMES:
        cs, cm = f"{m}_std", f"{m}_mean"
        if cs not in work.columns or cm not in work.columns:
            continue
        sub = work[[cs, cm]].dropna()
        if len(sub) < 8:
            continue
        if sub[cs].std(ddof=1) < 1e-12 or sub[cm].std(ddof=1) < 1e-12:
            rows.append({"metric": m, "n": len(sub), "spearman_r": np.nan, "p": np.nan, "note": "zero_variance"})
            continue
        r, p = stats.spearmanr(sub[cs], sub[cm])
        rows.append({"metric": m, "n": len(sub), "spearman_r": float(r), "p": float(p), "note": ""})
    return pd.DataFrame(rows)


def prepare_feature_table(
    df: pd.DataFrame,
    *,
    id_col: str = "subject_id",
    skip_ids: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    aggregate: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], List[str], Optional[int]]:
    """Helper."""
    work = df.copy()
    if skip_ids:
        work = work[~work[id_col].isin(skip_ids)]

    feats = list(feature_cols or MANIFOLD_COLS)
    feats = [c for c in feats if c in work.columns]
    n_rows_before = len(work) if aggregate else None

    if aggregate == "subject_mean":
        num_cols = feats + [c for c in work.columns if c not in (id_col, *feats)]
        num_cols = [c for c in num_cols if c in work.columns and pd.api.types.is_numeric_dtype(work[c])]
        work = work.groupby(id_col, as_index=False)[num_cols].mean(numeric_only=True)

    sub = work[[id_col] + feats].copy()
    keep = [c for c in feats if sub[c].std(ddof=1) > 1e-12]
    dropped = sorted(set(feats) - set(keep))
    return sub[[id_col] + keep], keep, dropped, n_rows_before


def run_manifold_fa(
    feature_df: pd.DataFrame,
    feat_cols: List[str],
    *,
    id_col: str = "subject_id",
    n_factors: int = 3,
    rotation: str = "varimax",
    method: str = "minres",
) -> Dict:
    n = len(feature_df)
    n_factors = min(n_factors, len(feat_cols) - 1, n - 2)
    if n_factors < 1:
        raise ValueError(f"")
    X = StandardScaler().fit_transform(feature_df[feat_cols].values.astype(float))

    kmo_all, kmo_model = calculate_kmo(X)
    chi2, p_bart = calculate_bartlett_sphericity(X)

    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
    fa.fit(X)

    fac_names = [f"F{i + 1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=feat_cols, columns=fac_names)
    scores = pd.DataFrame(fa.transform(X), columns=fac_names)
    scores.insert(0, id_col, feature_df[id_col].values)

    variance = fa.get_factor_variance()
    factor_detail = []
    for i, fac in enumerate(fac_names):
        factor_detail.append(
            {
                "factor": fac,
                "variance_explained": float(variance[1][i]),
                "salient_vars": _salient_vars(loadings, fac),
            }
        )

    return {
        "n_units": n,
        "n_features_in": len(feat_cols),
        "kmo_overall": float(kmo_model),
        "kmo_per_item": dict(zip(feat_cols, map(float, kmo_all))),
        "bartlett_chi2": float(chi2),
        "bartlett_p": float(p_bart),
        "rotation": rotation,
        "method": method,
        "n_factors": n_factors,
        "variance_proportion": variance[1].tolist(),
        "variance_cumulative": variance[2].tolist(),
        "loadings": loadings,
        "scores": scores,
        "communalities": pd.Series(fa.get_communalities(), index=feat_cols, name="communality"),
        "factor_detail": factor_detail,
    }


def correlate_factor_scores(
    scores: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    outcomes: Sequence[str],
    *,
    id_col: str = "subject_id",
) -> pd.DataFrame:
    merged = scores.merge(outcomes_df, on=id_col, how="left")
    rows = []
    fac_cols = [c for c in scores.columns if c.startswith("F")]
    for outcome in outcomes:
        if outcome not in merged.columns:
            continue
        for fac in fac_cols:
            sub = merged[[fac, outcome]].dropna()
            if len(sub) < 8:
                continue
            r, p = stats.spearmanr(sub[fac], sub[outcome])
            rows.append(
                {
                    "outcome": outcome,
                    "factor": fac,
                    "n": int(len(sub)),
                    "spearman_r": float(r),
                    "p": float(p),
                }
            )
    return pd.DataFrame(rows)


def plot_loadings_heatmap(loadings: pd.DataFrame, title: str, out_path: str) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.28 * len(loadings.index) + 2)), dpi=300)
    data = loadings.values
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(loadings.columns)))
    ax.set_xticklabels(loadings.columns, fontsize=10)
    ax.set_yticks(range(len(loadings.index)))
    ax.set_yticklabels(loadings.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=6.5, color="#111")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Loading")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_fa_markdown(
    res: Dict,
    *,
    cohort_label: str,
    metrics_csv: str,
    md_path: str,
    extra_lines: Optional[List[str]] = None,
    outcome_corr: Optional[pd.DataFrame] = None,
    dropped: Optional[List[str]] = None,
    aggregate_note: str = "",
) -> None:
    L: pd.DataFrame = res["loadings"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        f"",
        "",
        f"",
        f"",
        f"",
    ]
    if dropped:
        lines.append(f"- ：{', '.join(dropped) if dropped else '()'}")
    lines += [
        f"",
        "",
        "",
        "",
        f"- KMO = {res['kmo_overall']:.3f}",
        f"- Bartlett χ² = {res['bartlett_chi2']:.1f}, p = {res['bartlett_p']:.2e}",
        "",
        "",
        "",
        "",
        "|------|----------|------|-------------------|",
    ]
    cum = res["variance_cumulative"]
    for i, det in enumerate(res["factor_detail"]):
        sal = ", ".join(det["salient_vars"])
        lines.append(
            f"| {det['factor']} | {det['variance_explained']*100:.1f}% | {cum[i]*100:.1f}% | {sal} |"
        )

    lines += [
        "",
        "",
        "",
        "" + " | ".join(L.columns) + " |",
        "|------|" + "|".join(["------"] * len(L.columns)) + "|",
    ]
    for var in L.index:
        cells = []
        for fac in L.columns:
            v = L.loc[var, fac]
            mark = "**" if abs(v) >= LOADING_THRESHOLD else ""
            cells.append(f"{mark}{v:+.2f}{mark}")
        lines.append(f"| {var} | " + " | ".join(cells) + " |")

    if outcome_corr is not None and not outcome_corr.empty:
        lines += ["", "", ""]
        lines.append("|  |  | n | r | p |")
        lines.append("|------|------|---|---|---|")
        for _, row in outcome_corr.sort_values("p").iterrows():
            lines.append(
                f"| {row['factor']} | {row['outcome']} | {int(row['n'])} | "
                f"{row['spearman_r']:+.3f} | {row['p']:.4f} |"
            )

    if extra_lines:
        lines.extend(extra_lines)
    lines.append("")
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")


def save_fa_bundle(
    res: Dict,
    *,
    cohort_key: str,
    out_dir: str,
    metrics_csv: str,
    cohort_label: str,
    dropped: List[str],
    aggregate_note: str = "",
    outcome_corr: Optional[pd.DataFrame] = None,
    meta: Optional[Dict] = None,
) -> Dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{cohort_key}_fa"
    paths = {
        "loadings_csv": os.path.join(out_dir, f"{prefix}_loadings_{ts}.csv"),
        "scores_csv": os.path.join(out_dir, f"{prefix}_scores_{ts}.csv"),
        "json": os.path.join(out_dir, f"{prefix}_{ts}.json"),
        "md": os.path.join(out_dir, f""),
        "png": os.path.join(out_dir, f"{prefix}_loadings_{ts}.png"),
    }
    os.makedirs(out_dir, exist_ok=True)

    res["loadings"].to_csv(paths["loadings_csv"])
    res["scores"].to_csv(paths["scores_csv"], index=False)
    plot_loadings_heatmap(
        res["loadings"],
        f"{cohort_label} EFA loadings (varimax, 3 factors)",
        paths["png"],
    )
    write_fa_markdown(
        res,
        cohort_label=cohort_label,
        metrics_csv=metrics_csv,
        md_path=paths["md"],
        outcome_corr=outcome_corr,
        dropped=dropped,
        aggregate_note=aggregate_note,
    )

    summary = {
        "cohort": cohort_key,
        "cohort_label": cohort_label,
        "metrics_csv": metrics_csv,
        "timestamp": ts,
        "dropped_zero_variance": dropped,
        "aggregate_note": aggregate_note,
        **{k: v for k, v in res.items() if k not in ("loadings", "scores", "communalities")},
        "loadings": res["loadings"].to_dict(),
        "communalities": res["communalities"].to_dict(),
        "outcome_correlations": outcome_corr.to_dict(orient="records") if outcome_corr is not None and not outcome_corr.empty else [],
    }
    if meta:
        summary["meta"] = meta
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return paths
