"""Bidslab multi-night Apple Watch pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_BIDSLAB = Path(__file__).resolve().parent
_CPS = _BIDSLAB.parent / "cps_pilot"
for p in (_BIDSLAB, _CPS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bidslab_outcomes import compute_outcomes  # noqa: E402
from config_bidslab import (  # noqa: E402
    BIDSLAB_ROOT,
    NIGHT_CACHE,
    OUTPUT_DIR,
    OUTCOMES,
    SAMPLES_PER_EPOCH,
    UMAP_FRAME_DIR,
)
from fixed_umap import (  # noqa: E402
    analyze_fixed_frame,
    fit_umap_frame,
    load_frame,
    metric_keys,
    save_frame,
)
from load_night import (  # noqa: E402
    build_epoch_watch,
    epoch_watch_to_embedding,
    list_nights,
    list_subjects,
    load_hr,
    load_labels,
    load_motion,
)


def _night_paths(root: Path, subject: str, night: int):
    nd = root / subject / str(night)
    return nd / "labels.mat", nd / "motion.csv", nd / "hr.csv"


def _process_night(root: Path, subject: str, night: int, use_cache: bool = True) -> dict | None:
    cache_path = Path(NIGHT_CACHE) / f"{subject}_night{night}.npz"
    if use_cache and cache_path.is_file():
        z = np.load(cache_path, allow_pickle=True)
        emb = z["embedding"]
        if float(np.std(emb)) > 0:
            return {
                "subject_id": subject,
                "night": night,
                "embedding": emb,
                "outcomes": dict(z["outcomes"].item()) if "outcomes" in z else {},
                "n_epochs": int(z["n_epochs"]),
            }

    labels_p, motion_p, hr_p = _night_paths(root, subject, night)
    if not all(p.is_file() for p in (labels_p, motion_p, hr_p)):
        return None

    rec, _dreem, expert = load_labels(labels_p)
    out = compute_outcomes(expert)
    if out is None:
        return None

    mt, mxyz = load_motion(motion_p)
    ht, hb = load_hr(hr_p)
    watch = build_epoch_watch(rec, expert, mt, mxyz, ht, hb, SAMPLES_PER_EPOCH)
    emb = epoch_watch_to_embedding(watch)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    if len(emb) < 15:
        return None

    os.makedirs(NIGHT_CACHE, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embedding=emb,
        outcomes=out,
        n_epochs=len(expert),
    )
    return {
        "subject_id": subject,
        "night": night,
        "embedding": emb,
        "outcomes": out,
        "n_epochs": len(expert),
    }


def run(max_subjects: int = 0, skip_cache: bool = False, compare_refit: bool = True) -> pd.DataFrame:
    root = Path(BIDSLAB_ROOT)
    subjects = list_subjects(root)
    if max_subjects > 0:
        subjects = subjects[:max_subjects]

    rows = []
    refit_rows = []
    n_ok, n_fail = 0, 0

    for si, subject in enumerate(subjects, 1):
        nights = list_nights(root / subject)
        if not nights:
            continue

        frame_path = Path(UMAP_FRAME_DIR) / f"{subject}_night{nights[0]}.pkl"
        frame = load_frame(frame_path)

        for ni, night in enumerate(nights):
            pack = _process_night(root, subject, night, use_cache=not skip_cache)
            if pack is None:
                n_fail += 1
                continue

            emb = pack["embedding"]
            is_first = night == nights[0]

            if is_first or frame is None or frame.get("projector") is None:
                frame = fit_umap_frame(emb)
                if frame is not None:
                    save_frame(frame_path, frame)

            if frame is None:
                n_fail += 1
                continue

            try:
                metrics, _ = analyze_fixed_frame(emb, frame, refit_nightly=False)
            except Exception as exc:
                print(f"  WARN {subject} night{night}: {exc}", flush=True)
                n_fail += 1
                continue
            row = {
                "subject_id": subject,
                "night": night,
                "night_index": ni + 1,
                "is_first_night": is_first,
                "n_epochs": pack["n_epochs"],
                **metrics,
                **pack["outcomes"],
                "umap_mode": "fixed_night1",
            }
            rows.append(row)
            n_ok += 1

            if compare_refit:
                m_refit, _ = analyze_fixed_frame(emb, frame, refit_nightly=True)
                refit_rows.append(
                    {
                        "subject_id": subject,
                        "night": night,
                        "is_first_night": is_first,
                        **{f"{k}_refit": v for k, v in m_refit.items()},
                        **{k: row[k] for k in pack["outcomes"]},
                    }
                )

        if si % 5 == 0 or si == len(subjects):
            print(f"  [{si}/{len(subjects)}] nights_ok={n_ok} fail={n_fail}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"metrics_bidslab_fixed_{ts}.csv")
    df.to_csv(csv_path, index=False)

    corr = _correlate_outcomes(df)
    report = {
        "timestamp": ts,
        "n_subjects": df["subject_id"].nunique() if not df.empty else 0,
        "n_nights": len(df),
        "umap_policy": "fit on first night per subject; transform subsequent nights",
        "embedding": "watch IHR+|ACC|+dIHR epoch pooled to 192d",
        "csv": csv_path,
        "correlations_fixed": corr,
    }

    if compare_refit and refit_rows:
        df_r = pd.DataFrame(refit_rows)
        corr_refit = _correlate_outcomes_refit(df, df_r)
        report["correlations_nightly_refit"] = corr_refit

    json_path = os.path.join(OUTPUT_DIR, f"metrics_bidslab_fixed_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(OUTPUT_DIR, f"")
    _write_md(md_path, ts, df, corr, report.get("correlations_nightly_refit"))
    print(f"\n: {csv_path}\n      {md_path}")
    return df


def _correlate_outcomes(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    feats = [c for c in metric_keys() if c in df.columns]
    rows = []
    for target in OUTCOMES:
        for feat in feats:
            sub = df[[feat, target]].dropna()
            if len(sub) < 20:
                continue
            r, p = stats.spearmanr(sub[feat], sub[target])
            rows.append({"target": target, "feature": feat, "r": float(r), "p": float(p), "n": len(sub)})
    rows.sort(key=lambda x: abs(x["r"]), reverse=True)
    return rows


def _correlate_outcomes_refit(df: pd.DataFrame, df_r: pd.DataFrame) -> list:
    merged = df.merge(
        df_r[["subject_id", "night"] + [c for c in df_r.columns if c.endswith("_refit")]],
        on=["subject_id", "night"],
        how="inner",
    )
    rows = []
    for target in OUTCOMES:
        for feat in metric_keys():
            col = f"{feat}_refit"
            if col not in merged.columns:
                continue
            sub = merged[[col, target]].dropna()
            if len(sub) < 20:
                continue
            r, p = stats.spearmanr(sub[col], sub[target])
            rows.append({"target": target, "feature": feat, "r": float(r), "p": float(p), "n": len(sub)})
    rows.sort(key=lambda x: abs(x["r"]), reverse=True)
    return rows


def _write_md(path, ts, df, corr, corr_refit):
    lines = [
        f"",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"",
        "",
        "",
        "",
    ]
    for target in OUTCOMES:
        sub = [c for c in corr if c["target"] == target][:8]
        if not sub:
            continue
        lines.append(f"### {target}")
        lines.append("")
        lines.append("|  | n | r | p |")
        lines.append("|------|---|---|---|")
        for row in sub:
            lines.append(f"| {row['feature']} | {row['n']} | {row['r']:+.3f} | {row['p']:.4f} |")
        lines.append("")

    if corr_refit:
        lines.append("## ： refit UMAP（）")
        lines.append("")
        for target in OUTCOMES:
            sub = [c for c in corr_refit if c["target"] == target][:5]
            if not sub:
                continue
            lines.append(f"### {target}")
            for row in sub:
                lines.append(f"- {row['feature']}: r={row['r']:+.3f} p={row['p']:.4f}")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _clear_dir(path: Path) -> None:
    import shutil

    if not path.is_dir():
        return

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, 0o666)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_onerror)
    path.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Bidslab  UMAP ")
    ap.add_argument("--max_subjects", type=int, default=0)
    ap.add_argument("--skip_cache", action="store_true")
    ap.add_argument("--no_refit_compare", action="store_true")
    ap.add_argument(
        "--force_refit_frames",
        action="store_true",
        help=" UMAP frame， fit",
    )
    ap.add_argument(
        "--clear_night_cache",
        action="store_true",
        help=" nights ",
    )
    args = ap.parse_args()
    if args.force_refit_frames:
        _clear_dir(Path(UMAP_FRAME_DIR))
    if args.clear_night_cache:
        _clear_dir(Path(NIGHT_CACHE))
    run(
        max_subjects=args.max_subjects,
        skip_cache=args.skip_cache,
        compare_refit=not args.no_refit_compare,
    )


if __name__ == "__main__":
    main()
