"""Train embedding-level wearable distillation."""
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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

_DREAMT = os.path.dirname(os.path.abspath(__file__))
_CPS = os.path.join(os.path.dirname(_DREAMT), "cps_pilot")
for p in (_DREAMT, _CPS):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_dreamt import (  # noqa: E402
    DISTILL_OUTPUT_DIR,
    PARTICIPANT_CSV,
    TEACHER_EMB_CACHE,
)
from physical_dynamics_metrics import ALL_METRIC_NAMES, analyze_manifold_extended  # noqa: E402
from preprocess_dreamt import list_psg_subjects  # noqa: E402
from wearable_dataset import align_teacher_wearable, build_wearable_cache, extract_wearable_epochs  # noqa: E402
from wearable_student import WearableSleepFM, cosine_distill_loss  # noqa: E402

FOLDS_PKL = os.path.join(os.path.dirname(DISTILL_OUTPUT_DIR), "distill", "folds", "dreamt_folds.pkl")
SEED = 2481757


class EpochDistillDataset(Dataset):
    def __init__(self, pairs: List[Tuple[np.ndarray, np.ndarray]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        x, z = self.pairs[idx]
        return torch.from_numpy(x), torch.from_numpy(z)


def _load_teacher() -> Dict[str, np.ndarray]:
    with open(TEACHER_EMB_CACHE, "rb") as f:
        return pickle.load(f)


def _build_pairs(sids: List[str], teacher: Dict[str, np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for sid in sids:
        if sid not in teacher:
            continue
        w = extract_wearable_epochs(sid)
        if w is None:
            continue
        w, t = align_teacher_wearable(sid, teacher, w)
        for i in range(len(t)):
            pairs.append((w[i], t[i].astype(np.float32)))
    return pairs


def _make_folds(sids: List[str], n_folds: int = 5) -> Dict[int, dict]:
    if os.path.isfile(FOLDS_PKL):
        with open(FOLDS_PKL, "rb") as f:
            return pickle.load(f)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = {}
    arr = np.array(sorted(sids))
    for fid, (tr, te) in enumerate(kf.split(arr)):
        test = arr[te].tolist()
        tv = arr[tr].tolist()
        n_val = max(1, len(tv) // 5)
        folds[fid] = {"train": tv[n_val:], "val": tv[:n_val], "test": test}
    os.makedirs(os.path.dirname(FOLDS_PKL), exist_ok=True)
    with open(FOLDS_PKL, "wb") as f:
        pickle.dump(folds, f)
    return folds


@torch.no_grad()
def _eval_cosine(model: WearableSleepFM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    sims = []
    for x, z in loader:
        x, z = x.to(device), z.to(device)
        zs = model(x)
        zt = F.normalize(z, dim=1)
        sims.append((zs * zt).sum(dim=1).cpu().numpy())
    return float(np.concatenate(sims).mean()) if sims else float("nan")


def _train_fold(
    fold_id: int,
    folds: dict,
    teacher: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    train_pairs = _build_pairs(folds[fold_id]["train"], teacher)
    val_pairs = _build_pairs(folds[fold_id]["val"], teacher)
    train_loader = DataLoader(EpochDistillDataset(train_pairs), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(EpochDistillDataset(val_pairs), batch_size=batch_size, shuffle=False)

    model = WearableSleepFM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_sim = -1.0
    best_state = None
    history = []

    for ep in range(epochs):
        model.train()
        losses = []
        for x, z in train_loader:
            x, z = x.to(device), z.to(device)
            opt.zero_grad()
            zs = model(x)
            loss = cosine_distill_loss(zs, z)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_sim = _eval_cosine(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_cosine": val_sim})
        print(f"  fold{fold_id} ep{ep+1}/{epochs}  loss={np.mean(losses):.4f}  val_cos={val_sim:.4f}")
        if val_sim > best_sim:
            best_sim = val_sim
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    ckpt_dir = os.path.join(DISTILL_OUTPUT_DIR, f"wearable_student_fold{fold_id}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "val_cosine": best_sim}, os.path.join(ckpt_dir, "best.pt"))

    return {"fold": fold_id, "best_val_cosine": best_sim, "n_train": len(train_pairs), "n_val": len(val_pairs), "history": history}


@torch.no_grad()
def _extract_student_embeddings(
    model: WearableSleepFM,
    sids: List[str],
    teacher: Dict[str, np.ndarray],
    device: torch.device,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    model.eval()
    out: Dict[str, np.ndarray] = {}
    for sid in sids:
        w = extract_wearable_epochs(sid)
        if w is None or sid not in teacher:
            continue
        w, _ = align_teacher_wearable(sid, teacher, w)
        chunks = []
        bs = 128
        for i in range(0, len(w), bs):
            x = torch.from_numpy(w[i : i + bs]).to(device)
            chunks.append(model(x, normalize=normalize).cpu().numpy())
        out[sid] = np.concatenate(chunks, axis=0)
    return out


def _correlate_manifold(metrics_df, outcomes=("AHI", "ARI")) -> List[dict]:
    import pandas as pd
    rows = []
    feat_cols = [f"{m}_{s}" for m in ALL_METRIC_NAMES for s in ("std", "mean")]
    for target in outcomes:
        for feat in feat_cols:
            sub = metrics_df[[feat, target]].dropna()
            if len(sub) < 10:
                continue
            r, p = stats.spearmanr(sub[feat], sub[target])
            rows.append({"target": target, "feature": feat, "r": float(r), "p": float(p), "n": len(sub)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="SleepFM→ 128d ")
    parser.add_argument("--fold", type=int, default=0, help="-1=")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--build_cache", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    sids = list_psg_subjects()
    if args.build_cache:
        build_wearable_cache(sids)

    teacher = _load_teacher()
    folds = _make_folds([s for s in sids if s in teacher])

    fold_ids = list(folds.keys()) if args.fold < 0 else [args.fold]
    results = []
    for fid in fold_ids:
        print(f"\n=== Fold {fid} ===")
        results.append(_train_fold(fid, folds, teacher, device, args.epochs, args.batch_size, args.lr))

    fid = fold_ids[0]
    ckpt = torch.load(os.path.join(DISTILL_OUTPUT_DIR, f"wearable_student_fold{fid}", "best.pt"), map_location=device)
    model = WearableSleepFM().to(device)
    model.load_state_dict(ckpt["state_dict"])

    test_sids = folds[fid]["test"]
    student_emb = _extract_student_embeddings(model, test_sids, teacher, device)
    metric_rows = []
    for sid, emb in student_emb.items():
        m = analyze_manifold_extended(emb)
        metric_rows.append({"subject_id": sid, **m})
    import pandas as pd
    metrics_df = pd.DataFrame(metric_rows)
    pinfo = pd.read_csv(PARTICIPANT_CSV).rename(columns={"Arousal Index": "ARI", "SID": "subject_id"})
    metrics_df = metrics_df.merge(pinfo[["subject_id", "AHI", "ARI"]], on="subject_id", how="left")
    corr = _correlate_manifold(metrics_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(DISTILL_OUTPUT_DIR, exist_ok=True)
    report = {"timestamp": ts, "method": "cosine_embedding_distill", "fold_results": results, "test_manifold_correlations": corr}
    out_json = os.path.join(DISTILL_OUTPUT_DIR, f"wearable_distill_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    metrics_df.to_csv(os.path.join(DISTILL_OUTPUT_DIR, f"wearable_student_metrics_fold{fid}_{ts}.csv"), index=False)
    print(f"\n: {out_json}")
    if corr:
        top = sorted(corr, key=lambda x: abs(x["r"]), reverse=True)[:5]
        print(" vs AHI/ARI (Top):")
        for row in top:
            print(f"  {row['feature']} ~ {row['target']}: r={row['r']:+.3f} p={row['p']:.4f}")


if __name__ == "__main__":
    main()
