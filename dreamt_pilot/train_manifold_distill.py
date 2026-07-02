"""Train manifold-level distillation."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Dataset

_DREAMT = os.path.dirname(os.path.abspath(__file__))
_CPS = os.path.join(os.path.dirname(_DREAMT), "cps_pilot")
for p in (_DREAMT, _CPS):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_dreamt import (  # noqa: E402
    DISTILL_OUTPUT_DIR,
    OUTPUT_DIR,
    OUTCOMES,
    PARTICIPANT_CSV,
    PARTIAL_COVARIATES,
    TEACHER_EMB_CACHE,
)
from dreamt_sleep_outcomes import compute_sleep_outcomes  # noqa: E402
from analyze_mfi_partial import rank_residual  # noqa: E402
from manifold_distill_torch import TARGET_KEYS, differentiable_manifold_vector  # noqa: E402
from physical_dynamics_metrics import ALL_METRIC_NAMES, DISTILL_METRIC_NAMES, analyze_manifold_extended  # noqa: E402
from preprocess_dreamt import list_psg_subjects  # noqa: E402
from train_wearable_distill import (  # noqa: E402
    FOLDS_PKL,
    SEED,
    _extract_student_embeddings,
    _load_teacher,
    _make_folds,
)
from wearable_dataset import align_teacher_wearable, extract_wearable_epochs  # noqa: E402
from wearable_student import WearableSleepFM  # noqa: E402

TEACHER_METRIC_CACHE = os.path.join(DISTILL_OUTPUT_DIR, "cache", "teacher_manifold_metrics.pkl")


class SubjectDataset(Dataset):
    def __init__(self, sids: List[str]):
        self.sids = sids

    def __len__(self) -> int:
        return len(self.sids)

    def __getitem__(self, idx: int) -> str:
        return self.sids[idx]


def _metric_keys() -> List[str]:
    return [f"{m}_{s}" for m in DISTILL_METRIC_NAMES for s in ("std", "mean")]


def _teacher_metrics_from_emb(teacher: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    keys = _metric_keys()
    for sid, emb in teacher.items():
        m = analyze_manifold_extended(emb)
        out[sid] = {k: float(m.get(k, float("nan"))) for k in keys}
    return out


def _load_teacher_metrics(teacher: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    import pandas as pd

    os.makedirs(os.path.dirname(TEACHER_METRIC_CACHE), exist_ok=True)
    if os.path.isfile(TEACHER_METRIC_CACHE):
        with open(TEACHER_METRIC_CACHE, "rb") as f:
            cached = pickle.load(f)
        if set(cached.keys()) >= set(teacher.keys()):
            return cached

    keys = _metric_keys()
    csv_candidates = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("metrics_dreamt_") and f.endswith(".csv")
    )
    if csv_candidates:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, csv_candidates[-1]))
        df = df.rename(columns={"SID": "subject_id"})
        cached = {}
        for _, row in df.iterrows():
            sid = str(row["subject_id"])
            cached[sid] = {k: float(row[k]) for k in keys if k in row}
        if len(cached) >= len(teacher) * 0.9:
            with open(TEACHER_METRIC_CACHE, "wb") as f:
                pickle.dump(cached, f)
            print(f" CSV ({len(cached)} )", flush=True)
            return cached

    print(" (UMAP) ...", flush=True)
    cached = _teacher_metrics_from_emb(teacher)
    with open(TEACHER_METRIC_CACHE, "wb") as f:
        pickle.dump(cached, f)
    return cached


def _stats_from_subjects(
    teacher_metrics: Dict[str, Dict[str, float]], sids: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = _metric_keys()
    mat = np.array([[teacher_metrics[s][k] for k in keys] for s in sids], dtype=np.float64)
    mat_t = np.sign(mat) * np.log1p(np.abs(mat))
    mu = np.nanmean(mat_t, axis=0)
    sd = np.nanstd(mat_t, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    weights = 1.0 / sd
    weights = weights / weights.mean()
    return mu, sd, weights.astype(np.float32)


def _forward_subject(
    model: WearableSleepFM, sid: str, device: torch.device, chunk: int = 128
) -> torch.Tensor:
    w = extract_wearable_epochs(sid)
    if w is None:
        return torch.empty(0, 128, device=device)
    parts = []
    for i in range(0, len(w), chunk):
        x = torch.from_numpy(w[i : i + chunk]).to(device)
        parts.append(model(x, normalize=False))
    z = torch.cat(parts, dim=0)
    if len(z) < 60 or not torch.isfinite(z).all():
        return torch.empty(0, 128, device=device)
    return torch.clamp(z, -20.0, 20.0)


def _transform_target(target: np.ndarray) -> np.ndarray:
    return np.sign(target) * np.log1p(np.abs(target))


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _manifold_loss(
    pred: torch.Tensor,
    target: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    device: torch.device,
    metric_weights: np.ndarray | None = None,
) -> torch.Tensor:
    """Helper."""
    p = _signed_log1p(pred)
    tgt_t = _transform_target(target)
    t = torch.from_numpy((tgt_t - mu) / sd).to(device=device, dtype=pred.dtype)
    p = (p - torch.from_numpy(mu).to(device, dtype=pred.dtype)) / torch.from_numpy(sd).to(
        device, dtype=pred.dtype
    )
    mask = torch.isfinite(t) & torch.isfinite(p)
    if mask.sum() == 0:
        return pred.new_tensor(float("nan"))
    diff = p[mask] - t[mask]
    if metric_weights is not None:
        w = torch.from_numpy(metric_weights).to(device, dtype=pred.dtype)[mask]
        diff = diff * w
    return F.smooth_l1_loss(diff, torch.zeros_like(diff))


@torch.no_grad()
def _eval_subject_metrics(
    model: WearableSleepFM,
    sids: List[str],
    teacher_metrics: Dict[str, Dict[str, float]],
    device: torch.device,
) -> Tuple[float, float]:
    """Helper."""
    keys = _metric_keys()
    st_pairs, umap_pairs = [], []
    for sid in sids:
        emb = _forward_subject(model, sid, device).cpu().numpy()
        if len(emb) < 60 or not np.isfinite(emb).all():
            continue
        pred_umap = analyze_manifold_extended(emb)
        pred_torch = differentiable_manifold_vector(
            torch.from_numpy(emb).to(device)
        ).cpu().numpy()
        tgt = teacher_metrics[sid]
        for k in keys:
            tv, uv = tgt.get(k, float("nan")), pred_umap.get(k, float("nan"))
            pv = float(pred_torch[keys.index(k)])
            if np.isfinite(tv) and np.isfinite(pv):
                st_pairs.append((tv, pv))
            if np.isfinite(tv) and np.isfinite(uv):
                umap_pairs.append((tv, uv))
    if len(st_pairs) < 10:
        return float("nan"), float("nan")
    st_r = stats.pearsonr(*zip(*st_pairs)).statistic
    umap_r = stats.pearsonr(*zip(*umap_pairs)).statistic if len(umap_pairs) >= 10 else float("nan")
    return float(st_r), float(umap_r)


def _train_fold(
    fold_id: int,
    folds: dict,
    teacher_metrics: Dict[str, Dict[str, float]],
    mu: np.ndarray,
    sd: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int = 2,
    min_epochs: int = 2,
    pca_ridge_eps: float = 1e-6,
) -> dict:
    train_sids = folds[fold_id]["train"]
    val_sids = folds[fold_id]["val"]
    train_loader = DataLoader(SubjectDataset(train_sids), batch_size=1, shuffle=True)

    model = WearableSleepFM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    keys = _metric_keys()
    best_val = float("inf")
    best_state = None
    history = []
    no_improve = 0
    nan_epochs = 0

    for ep in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            sid = batch[0] if isinstance(batch, (list, tuple)) else batch
            opt.zero_grad()
            z = _forward_subject(model, sid, device)
            if len(z) < 60:
                continue
            pred = differentiable_manifold_vector(z, ridge_eps=pca_ridge_eps)
            if not pred.requires_grad:
                continue
            tgt = np.array([teacher_metrics[sid][k] for k in keys], dtype=np.float32)
            loss = _manifold_loss(pred, tgt, mu, sd, device, weights)
            if not torch.isfinite(loss) or not loss.requires_grad:
                continue
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            except RuntimeError:
                opt.zero_grad()
                continue
            opt.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                if best_state:
                    model.load_state_dict(best_state)
                opt.zero_grad()
                continue
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        if not np.isfinite(train_loss):
            nan_epochs += 1
        else:
            nan_epochs = 0

        model.eval()
        val_losses = []
        with torch.no_grad():
            for sid in val_sids:
                z = _forward_subject(model, sid, device)
                if len(z) < 60:
                    continue
                pred = differentiable_manifold_vector(z, ridge_eps=pca_ridge_eps)
                tgt = np.array([teacher_metrics[sid][k] for k in keys], dtype=np.float32)
                val_losses.append(float(_manifold_loss(pred, tgt, mu, sd, device, weights).item()))

        val_mse = float(np.mean(val_losses)) if val_losses else float("nan")
        if ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == epochs:
            try:
                st_r, umap_r = _eval_subject_metrics(model, val_sids[:8], teacher_metrics, device)
            except ValueError:
                st_r, umap_r = float("nan"), float("nan")
        else:
            st_r, umap_r = float("nan"), float("nan")
        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_metric_pearson_torch": st_r,
            "val_metric_pearson_umap": umap_r,
        })
        print(
            f"  fold{fold_id} ep{ep+1}/{epochs}  train={train_loss:.4f}  "
            f"val_mse={val_mse:.4f}  val_r(torch)={st_r:.3f}  val_r(umap)={umap_r:.3f}",
            flush=True,
        )
        improved = np.isfinite(val_mse) and val_mse < best_val - 1e-8
        if improved:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if ep + 1 >= min_epochs and no_improve >= patience:
            print(f"  fold{fold_id} early stop @ ep{ep+1} (patience={patience})", flush=True)
            break
        if nan_epochs >= 2 and best_state is not None:
            print(f"  fold{fold_id} early stop @ ep{ep+1} (train NaN×{nan_epochs})", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    ckpt_dir = os.path.join(DISTILL_OUTPUT_DIR, f"manifold_student_fold{fold_id}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "val_mse": best_val}, os.path.join(ckpt_dir, "best.pt"))
    return {"fold": fold_id, "best_val_mse": best_val, "history": history}


def _correlate_manifold(
    metrics_df,
    outcomes=("AHI", "ARI", "TST_hours", "sleep_efficiency"),
    partial: bool = True,
) -> List[dict]:
    rows = []
    feat_cols = [f"{m}_{s}" for m in ALL_METRIC_NAMES for s in ("std", "mean")]
    covs = list(PARTIAL_COVARIATES)
    for target in outcomes:
        if target not in metrics_df.columns:
            continue
        for feat in feat_cols:
            if feat not in metrics_df.columns:
                continue
            cols = [feat, target] + (covs if partial else [])
            sub = metrics_df[cols].dropna()
            if len(sub) < 10:
                continue
            raw_r, raw_p = stats.spearmanr(sub[feat], sub[target])
            row = {
                "target": target,
                "feature": feat,
                "raw_r": float(raw_r),
                "raw_p": float(raw_p),
                "n": len(sub),
            }
            if partial and all(c in sub.columns for c in covs):
                yr = rank_residual(sub[target].values.astype(float), sub[covs].values.astype(float))
                xr = rank_residual(sub[feat].values.astype(float), sub[covs].values.astype(float))
                pr, pp = stats.pearsonr(yr, xr)
                row["partial_r"] = float(pr)
                row["partial_p"] = float(pp)
            rows.append(row)
    return rows


def _enrich_outcomes(metrics_df):
    import pandas as pd

    pinfo = pd.read_csv(PARTICIPANT_CSV).rename(
        columns={"Arousal Index": "ARI", "SID": "subject_id"}
    )
    pinfo["GENDER_M"] = (pinfo["GENDER"].astype(str).str.upper() == "M").astype(float)
    sleep_rows = []
    for sid in metrics_df["subject_id"]:
        s = compute_sleep_outcomes(sid) or {}
        sleep_rows.append({"subject_id": sid, **s})
    sleep_df = pd.DataFrame(sleep_rows)
    out = metrics_df.merge(
        pinfo[["subject_id", "AHI", "ARI", "AGE", "BMI", "GENDER_M"]],
        on="subject_id",
        how="left",
    )
    out = out.merge(
        sleep_df[["subject_id", "TST_hours", "sleep_efficiency"]],
        on="subject_id",
        how="left",
    )
    return out


def _pooled_test_eval(
    fold_ids: List[int],
    folds: dict,
    teacher: Dict[str, np.ndarray],
    teacher_metrics: Dict[str, Dict[str, float]],
    device: torch.device,
) -> tuple:
    import pandas as pd

    keys = _metric_keys()
    metric_rows, recovery_rows = [], []
    for fid in fold_ids:
        ckpt_path = os.path.join(DISTILL_OUTPUT_DIR, f"manifold_student_fold{fid}", "best.pt")
        if not os.path.isfile(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model = WearableSleepFM().to(device)
        model.load_state_dict(ckpt["state_dict"])
        test_sids = folds[fid]["test"]
        student_emb = _extract_student_embeddings(model, test_sids, teacher, device, normalize=False)
        for sid, emb in student_emb.items():
            m = analyze_manifold_extended(emb)
            metric_rows.append({"subject_id": sid, "fold": fid, **m})
            tgt = teacher_metrics[sid]
            for k in keys:
                if np.isfinite(tgt[k]) and np.isfinite(m.get(k, float("nan"))):
                    recovery_rows.append(
                        {"subject_id": sid, "fold": fid, "feature": k, "teacher": tgt[k], "student": m[k]}
                    )
    metrics_df = _enrich_outcomes(pd.DataFrame(metric_rows))
    corr = _correlate_manifold(metrics_df)
    recovery = []
    rec_df = pd.DataFrame(recovery_rows)
    if not rec_df.empty:
        for feat in keys:
            sub = rec_df[rec_df["feature"] == feat]
            if len(sub) < 10:
                continue
            r, p = stats.pearsonr(sub["teacher"], sub["student"])
            recovery.append({"feature": feat, "r": float(r), "p": float(p), "n": len(sub)})
    return metrics_df, corr, recovery


def _write_summary_md(
    path: str,
    ts: str,
    fold_results: List[dict],
    metrics_df,
    corr: List[dict],
    recovery: List[dict],
    args,
) -> None:
    n_sub = len(metrics_df)
    lines = [
        f"",
        "",
        "",
        "",
        "",
        "",
        "",
        f"",
        "",
        f"",
        "",
    ]
    if recovery:
        lines.append("### ↔")
        lines.append("")
        lines.append("|  | n | r | p |")
        lines.append("|------|---|---|---|")
        for row in sorted(recovery, key=lambda x: abs(x["r"]), reverse=True)[:10]:
            lines.append(f"| {row['feature']} | {row['n']} | {row['r']:+.3f} | {row['p']:.4f} |")
        lines.append("")
    for target in OUTCOMES:
        sub = [c for c in corr if c["target"] == target]
        if not sub:
            continue
        sig = [c for c in sub if c["raw_p"] < 0.05]
        lines.append(f"### {target}（raw p<0.05: {len(sig)}/{len(sub)}）")
        lines.append("")
        lines.append("|  | n | raw r | raw p | partial r | partial p |")
        lines.append("|------|---|-------|-------|-----------|-----------|")
        for row in sorted(sub, key=lambda x: abs(x["raw_r"]), reverse=True)[:10]:
            pr = row.get("partial_r", float("nan"))
            pp = row.get("partial_p", float("nan"))
            prs = f"{pr:+.3f}" if np.isfinite(pr) else "—"
            pps = f"{pp:.4f}" if np.isfinite(pp) else "—"
            lines.append(
                f"| {row['feature']} | {row['n']} | {row['raw_r']:+.3f} | {row['raw_p']:.4f} | {prs} | {pps} |"
            )
        lines.append("")
    lines.extend([
        "",
        "",
        "",
        "|------|----------------|",
        "| AHI | MTC_mean r≈+0.30 |",
        "| ARI | MFI_mean r≈-0.27 |",
        "| TST | MSV_mean r≈-0.21 |",
        "| sleep_efficiency | MSV_mean r≈-0.20 |",
        "",
        "",
        "",
        f"- `{path}`",
        f"- `output/dreamt/distill/manifold_student_pooled_{ts}.csv`",
        f"- `output/dreamt/distill/manifold_distill_{ts}.json`",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="： LCI/SMI/MFI/VTI + ")
    parser.add_argument("--fold", type=int, default=0, help="-1=")
    parser.add_argument("--epochs", type=int, default=4, help=" epoch； 3–4 ")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2, help=" loss ")
    parser.add_argument("--min_epochs", type=int, default=2, help="")
    parser.add_argument("--pca_ridge_eps", type=float, default=1e-6, help=" PCA  Ridge")
    parser.add_argument("--skip_folds", type=str, default="", help="，， 0")
    parser.add_argument("--rebuild_teacher_metrics", action="store_true")
    args = parser.parse_args()

    skip_folds = {int(x) for x in args.skip_folds.split(",") if x.strip().isdigit()}

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    if args.rebuild_teacher_metrics and os.path.isfile(TEACHER_METRIC_CACHE):
        os.remove(TEACHER_METRIC_CACHE)

    teacher = _load_teacher()
    teacher_metrics = _load_teacher_metrics(teacher)
    sids = [s for s in list_psg_subjects() if s in teacher_metrics]
    folds = _make_folds(sids)

    fold_ids = list(folds.keys()) if args.fold < 0 else [args.fold]
    results = []
    for fid in fold_ids:
        if fid in skip_folds and os.path.isfile(
            os.path.join(DISTILL_OUTPUT_DIR, f"manifold_student_fold{fid}", "best.pt")
        ):
            print(f"\n=== Fold {fid} （ checkpoint）===", flush=True)
            continue
        print(f"\n=== Fold {fid}  ===", flush=True)
        mu, sd, weights = _stats_from_subjects(teacher_metrics, folds[fid]["train"])
        results.append(
            _train_fold(
                fid, folds, teacher_metrics, mu, sd, weights, device,
                args.epochs, args.lr, args.patience, args.min_epochs, args.pca_ridge_eps,
            )
        )

    metrics_df, corr, recovery = _pooled_test_eval(fold_ids, folds, teacher, teacher_metrics, device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "method": "manifold_metric_distill",
        "note": "",
        "fold_results": results,
        "n_pooled_test": len(metrics_df),
        "test_teacher_student_recovery": recovery,
        "test_manifold_correlations": corr,
    }
    out_json = os.path.join(DISTILL_OUTPUT_DIR, f"manifold_distill_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    pooled_csv = os.path.join(DISTILL_OUTPUT_DIR, f"manifold_student_pooled_{ts}.csv")
    metrics_df.to_csv(pooled_csv, index=False)
    md_path = os.path.join(DISTILL_OUTPUT_DIR, f"")
    _write_summary_md(md_path, ts, results, metrics_df, corr, recovery, args)
    print(f"\n: {out_json}", flush=True)
    print(f"      {pooled_csv}", flush=True)
    print(f"      {md_path}", flush=True)

    if recovery:
        top = sorted(recovery, key=lambda x: abs(x["r"]), reverse=True)[:5]
        print("Pooled ↔  (Top):", flush=True)
        for row in top:
            print(f"  {row['feature']}: r={row['r']:+.3f} p={row['p']:.4f}", flush=True)
    if corr:
        for target in OUTCOMES:
            sub = sorted(
                [c for c in corr if c["target"] == target],
                key=lambda x: abs(x["raw_r"]),
                reverse=True,
            )[:3]
            if sub:
                print(f"Pooled vs {target} (Top):", flush=True)
                for row in sub:
                    print(
                        f"  {row['feature']}: raw r={row['raw_r']:+.3f} p={row['raw_p']:.4f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
