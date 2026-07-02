"""CPS end-to-end pipeline."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch

from config_cps import CPS_DATA_ROOT, MODEL_CHECKPOINT, OUTPUT_DIR
from embeddings import _load_sleepfm_models, extract_subject_embeddings
from manifold_metrics import PRIMARY_METRIC_NAMES, analyze_embedding
from preprocess_cps import preprocess_subjects


def _load_fold(path: str, n: int = 0) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    return ids[:n] if n > 0 else ids


def _diagnose_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Helper."""
    metrics = [m for m in PRIMARY_METRIC_NAMES if m in df.columns]
    if not metrics:
        metrics = ["LCI_std", "SMI_std", "MFI_std", "VTI_std"]
    flags = []
    for _, row in df.iterrows():
        z = {}
        for m in metrics:
            col = df[m].dropna()
            if len(col) < 3:
                z[m] = 0.0
                continue
            z[m] = abs((row[m] - col.mean()) / (col.std() + 1e-8))
        score = (
            z.get("LCI_std", 0) + z.get("MFI_std", 0)
            + max(0, -z.get("SMI_std", 0)) + max(0, -z.get("VTI_std", 0))
        )
        flags.append(score)
    df = df.copy()
    df["dynamics_anomaly_score"] = flags
    threshold = df["dynamics_anomaly_score"].mean() + df["dynamics_anomaly_score"].std()
    df["is_anomaly"] = df["dynamics_anomaly_score"] > threshold
    return df


def run_pipeline(
    sample_ids: List[str],
    output_dir: str = OUTPUT_DIR,
    checkpoint: str = MODEL_CHECKPOINT,
    batch_size: int = 32,
    max_subjects: int = 0,
    output_tag: str = "",
    skip_preprocess: bool = False,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    if max_subjects > 0:
        sample_ids = sample_ids[:max_subjects]

    if skip_preprocess:
        ok_ids = [sid for sid in sample_ids if os.path.isdir(os.path.join(output_dir, "X", sid))]
        print(f"[1/4] ， X : {len(ok_ids)}/{len(sample_ids)}")
    else:
        print(f"[1/4]  {len(sample_ids)}  ...")
        ok_ids = preprocess_subjects(sample_ids, output_dir, CPS_DATA_ROOT)
        print(f"      : {len(ok_ids)}/{len(sample_ids)}")

    rows = []
    embedding_mode = "unknown"
    models_tuple = None
    dev = None
    if checkpoint and os.path.isfile(checkpoint):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[2/4]  SleepFM checkpoint ({dev}) ...")
        models_tuple = _load_sleepfm_models(checkpoint, dev)

    for i, sid in enumerate(ok_ids, 1):
        x_dir = os.path.join(output_dir, "X", sid)
        emb, files, mode = extract_subject_embeddings(
            x_dir,
            batch_size=batch_size,
            checkpoint_path=checkpoint,
            models_tuple=models_tuple,
            dev=dev,
        )
        embedding_mode = mode
        metrics = analyze_embedding(emb)
        rows.append(
            {
                "subject_id": sid,
                "n_epochs": len(files),
                "embedding_mode": mode,
                **metrics,
            }
        )
        lci = metrics.get("LCI_std", float("nan"))
        smi = metrics.get("SMI_std", float("nan"))
        vti = metrics.get("VTI_std", float("nan"))
        nw = metrics.get("n_windows", 0)
        print(
            f"      [{i}/{len(ok_ids)}] {sid}: epochs={len(files)}, windows={int(nw)}, "
            f"LCI_std={lci:.4f}, SMI_std={smi:.4f}, VTI_std={vti:.4f}"
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] ")
        return df

    df = _diagnose_outliers(df)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{output_tag}_" if output_tag else ""
    csv_path = os.path.join(output_dir, f"metrics_{tag}{ts}.csv")
    df.to_csv(csv_path, index=False)

    report = {
        "timestamp": ts,
        "n_subjects_requested": len(sample_ids),
        "n_subjects_processed": len(ok_ids),
        "embedding_mode": embedding_mode,
        "checkpoint_used": os.path.isfile(checkpoint),
        "checkpoint_path": checkpoint,
        "metrics_summary": df[[c for c in PRIMARY_METRIC_NAMES if c in df.columns]].describe().to_dict(),
        "anomalies": df[df["is_anomaly"]]["subject_id"].tolist(),
        "output_csv": csv_path,
    }
    report_path = os.path.join(output_dir, f"report_{tag}{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[4/4] :")
    print(f"      CSV: {csv_path}")
    print(f"      JSON: {report_path}")
    if report["anomalies"]:
        print(f"      : {report['anomalies']}")
    else:
        print("      ")
    return df


def main():
    parser = argparse.ArgumentParser(description="CPS + SleepFM ")
    parser.add_argument("--max_subjects", type=int, default=0, help="（0=）")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--output_tag", type=str, default="umap_sliding", help="")
    parser.add_argument("--skip_preprocess", action="store_true", help="，+")
    parser.add_argument("--checkpoint", type=str, default=MODEL_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--fold",
        type=str,
        default=os.path.join(OUTPUT_DIR, "psqi_subjects.txt"),
        help=" ID ",
    )
    args = parser.parse_args()

    sample_ids = _load_fold(args.fold, n=args.max_subjects if args.max_subjects > 0 else 0)
    run_pipeline(
        sample_ids=sample_ids,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        max_subjects=0,
        output_tag=args.output_tag,
        skip_preprocess=args.skip_preprocess,
    )


if __name__ == "__main__":
    main()
