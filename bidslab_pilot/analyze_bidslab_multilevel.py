"""Bidslab pooled and mixed-model analyses."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

_B = Path(__file__).resolve().parent
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

from config_bidslab import OUTPUT_DIR, OUTCOMES

DEFAULT_CSV = Path(OUTPUT_DIR) / "metrics_bidslab_fixed_20260615_131155.csv"
ANALYSIS_DIR = Path(OUTPUT_DIR) / "analysis"

PRIMARY_FEATURES = {
    "TST_hours": ["MRR_std", "MRR_mean", "VTI_std", "MTE_mean"],
    "sleep_efficiency": ["MRR_std", "MTC_mean", "MFI_mean", "VTI_std"],
}
PLOT_FEATURE = "MRR_std"
PLOT_OUTCOME = "TST_hours"


def _latest_csv() -> Path:
    cands = sorted(Path(OUTPUT_DIR).glob("metrics_bidslab_fixed_*.csv"))
    return cands[-1] if cands else DEFAULT_CSV


def _load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"subject_id", "night", PLOT_FEATURE, "TST_hours", "sleep_efficiency"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"")
    return df.dropna(subset=["subject_id", PLOT_FEATURE, "TST_hours", "sleep_efficiency"])


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
            "random_intercept_var": float(result.cov_re.iloc[0, 0]),
            "aic": float(result.aic),
            "formula": f"{formula} + (1 | subject_id)",
        }
    except Exception as exc:
        return {"outcome": outcome, "feature": feature, "ok": False, "reason": str(exc)}


def run_between_subject(df: pd.DataFrame, outcome: str, feature: str) -> dict:
    subj = (
        df.groupby("subject_id", as_index=False)[[feature, outcome]]
        .mean(numeric_only=True)
        .dropna()
    )
    if len(subj) < 10:
        return {"outcome": outcome, "feature": feature, "ok": False, "reason": "insufficient subjects"}
    r, p = stats.spearmanr(subj[feature], subj[outcome])
    return {
        "outcome": outcome,
        "feature": feature,
        "ok": True,
        "n_subjects": int(len(subj)),
        "spearman_r": float(r),
        "pvalue": float(p),
    }


def run_pooled_spearman(df: pd.DataFrame, outcome: str, feature: str) -> dict:
    sub = df[[feature, outcome]].dropna()
    r, p = stats.spearmanr(sub[feature], sub[outcome])
    return {
        "outcome": outcome,
        "feature": feature,
        "n_nights": int(len(sub)),
        "spearman_r": float(r),
        "pvalue": float(p),
    }


def _subject_colors(subjects: list[str]) -> dict[str, tuple]:
    cmap = plt.colormaps.get_cmap("tab20").resampled(20)
    cmap2 = plt.colormaps.get_cmap("tab20b").resampled(20)
    colors = [cmap(i % 20) for i in range(20)] + [cmap2(i % 20) for i in range(20)]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    out = {}
    for i, sid in enumerate(subjects):
        out[sid] = (colors[i % len(colors)], markers[i % len(markers)])
    return out


def plot_multinight_scatter(
    df: pd.DataFrame,
    feature: str,
    outcome: str,
    out_path: Path,
    lmm: dict | None = None,
    between: dict | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    sub = df[["subject_id", "night", feature, outcome]].dropna().copy()
    subjects = sorted(sub["subject_id"].unique())
    style = _subject_colors(subjects)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300, facecolor="white")
    ax.set_facecolor("#fafafa")

    for sid, grp in sub.groupby("subject_id"):
        grp = grp.sort_values("night")
        color, marker = style[sid]
        x = grp[feature].values
        y = grp[outcome].values
        ax.plot(x, y, color=color, alpha=0.22, linewidth=0.9, zorder=1)
        ax.scatter(
            x,
            y,
            c=[color],
            marker=marker,
            s=38,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.45,
            zorder=2,
            label=sid,
        )

    if len(sub) >= 3:
        slope, intercept = np.polyfit(sub[feature], sub[outcome], 1)
        xs = np.linspace(sub[feature].min(), sub[feature].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#1e293b", linewidth=1.6, alpha=0.75, zorder=3)

    rho, p_pool = stats.spearmanr(sub[feature], sub[outcome])

    lines = [
        f"Pooled nights: n = {len(sub)}",
        f"Spearman r = {rho:+.3f}, p = {p_pool:.2e}",
    ]
    if lmm and lmm.get("ok"):
        lines.append(
            f"LMM coef = {lmm['coef']:+.4f} (p = {lmm['pvalue']:.2e}, "
            f"{lmm['n_subjects']} subjects)"
        )
    if between and between.get("ok"):
        lines.append(
            f"Between-subject: r = {between['spearman_r']:+.3f}, "
            f"p = {between['pvalue']:.4f} (n = {between['n_subjects']})"
        )

    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
    )

    ax.set_xlabel(f"{feature} (fixed UMAP manifold)", fontsize=11, fontweight="medium")
    ax.set_ylabel(f"{outcome}", fontsize=11, fontweight="medium")
    ax.set_title(
        f"{feature} vs {outcome}\n253 nights, 47 subjects (color = subject)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = []
    for sid in subjects[:12]:
        color, marker = style[sid]
        handles.append(
            mlines.Line2D(
                [], [], color=color, marker=marker, linestyle="None", markersize=5, label=sid
            )
        )
    if len(subjects) > 12:
        handles.append(mlines.Line2D([], [], color="none", label=f"... +{len(subjects) - 12} subjects"))
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=6.5,
        frameon=True,
        framealpha=0.92,
        edgecolor="#e2e8f0",
        title="Subject ID",
        title_fontsize=7,
        ncol=2,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_report(
    path: Path,
    ts: str,
    csv_path: Path,
    lmm_rows: list[dict],
    between_rows: list[dict],
    pooled_rows: list[dict],
    plot_path: Path,
) -> None:
    lines = [
        f"",
        "",
        f"",
        "",
        "",
        "",
        "",
        "",
        "",
        "|------|------|-----|-----|------|----|----|--------|",
    ]
    for row in lmm_rows:
        if not row.get("ok"):
            lines.append(f"| {row['outcome']} | {row['feature']} | — | — | — | — | — | {row.get('reason','fail')} |")
            continue
        lines.append(
            f"| {row['outcome']} | {row['feature']} | {row['n_nights']} | {row['n_subjects']} | "
            f"{row['coef']:+.4f} | {row['se']:.4f} | {row['pvalue']:.2e} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )

    lines += [
        "",
        "",
        "",
        "",
        "|------|------|-----|------------|---|",
    ]
    for row in between_rows:
        if not row.get("ok"):
            lines.append(f"| {row['outcome']} | {row['feature']} | — | — | {row.get('reason','fail')} |")
            continue
        lines.append(
            f"| {row['outcome']} | {row['feature']} | {row['n_subjects']} | "
            f"{row['spearman_r']:+.3f} | {row['pvalue']:.4f} |"
        )

    lines += [
        "",
        "",
        "",
        "",
        "|------|------|---|---|---|",
    ]
    for row in pooled_rows:
        lines.append(
            f"| {row['outcome']} | {row['feature']} | {row['n_nights']} | "
            f"{row['spearman_r']:+.3f} | {row['pvalue']:.2e} |"
        )

    lines += [
        "",
        "",
        "",
        f"![{PLOT_FEATURE} vs {PLOT_OUTCOME}]({plot_path.name})",
        "",
        f"",
        "",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Bidslab LMM + between-subject + scatter")
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else _latest_csv()
    df = _load_df(csv_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    lmm_rows = []
    between_rows = []
    pooled_rows = []

    for outcome, feats in PRIMARY_FEATURES.items():
        for feat in feats:
            if feat not in df.columns:
                continue
            lmm_rows.append(run_lmm(df, outcome, feat))
            between_rows.append(run_between_subject(df, outcome, feat))
            pooled_rows.append(run_pooled_spearman(df, outcome, feat))

    lmm_primary = run_lmm(df, PLOT_OUTCOME, PLOT_FEATURE)
    between_primary = run_between_subject(df, PLOT_OUTCOME, PLOT_FEATURE)

    plot_path = ANALYSIS_DIR / f"scatter_{PLOT_FEATURE}_vs_{PLOT_OUTCOME}_{ts}.png"
    plot_multinight_scatter(
        df,
        PLOT_FEATURE,
        PLOT_OUTCOME,
        plot_path,
        lmm=lmm_primary,
        between=between_primary,
    )

    report = {
        "timestamp": ts,
        "csv": str(csv_path),
        "n_nights": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()),
        "lmm": lmm_rows,
        "between_subject": between_rows,
        "pooled_spearman": pooled_rows,
        "plot": str(plot_path),
    }
    json_path = ANALYSIS_DIR / f"bidslab_multilevel_{ts}.json"
    md_path = ANALYSIS_DIR / f""
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_report(md_path, ts, csv_path, lmm_rows, between_rows, pooled_rows, plot_path)

    print(f": {md_path}")
    print(f"      {json_path}")
    print(f"      {plot_path}")
    if lmm_primary.get("ok"):
        print(
            f"LMM {PLOT_OUTCOME}~{PLOT_FEATURE}: coef={lmm_primary['coef']:+.4f}, "
            f"p={lmm_primary['pvalue']:.2e}"
        )
    if between_primary.get("ok"):
        print(
            f"Between-subject: r={between_primary['spearman_r']:+.3f}, "
            f"p={between_primary['pvalue']:.4f}"
        )


if __name__ == "__main__":
    main()
