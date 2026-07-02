"""APNEA end-to-end manifold pipeline."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

_APNEA = os.path.dirname(os.path.abspath(__file__))
_CPS = os.path.join(os.path.dirname(_APNEA), "cps_pilot")
for p in (_APNEA, _CPS):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_apnea import MIN_EPOCHS, MODEL_CHECKPOINT, ODI_ALIGNMENT_CSV, OUTPUT_DIR  # noqa: E402
from embeddings import _load_sleepfm_models, extract_subject_embeddings  # noqa: E402
from physical_dynamics_metrics import ALL_METRIC_NAMES, analyze_manifold_extended  # noqa: E402
from preprocess_apnea import list_apnea_patients, preprocess_all, preprocess_patient  # noqa: E402

EMB_CACHE = os.path.join(OUTPUT_DIR, "cache", "embeddings_128d.pkl")


def _load_outcomes() -> pd.DataFrame:
    df = pd.read_csv(ODI_ALIGNMENT_CSV)
    df = df.rename(columns={"patient_id": "subject_id"})
    df["ARI"] = df["true_ari_from_pt"].astype(float)
    df["AHI"] = df["ahi_l3_from_pt"].astype(float)
    return df[["subject_id", "ARI", "AHI", "recording_hours_from_rml", "n_windows_from_pt"]]


def _load_embeddings(sids: List[str], checkpoint: str) -> Dict[str, np.ndarray]:
    os.makedirs(os.path.dirname(EMB_CACHE), exist_ok=True)
    cached: Dict[str, np.ndarray] = {}
    if os.path.isfile(EMB_CACHE):
        with open(EMB_CACHE, "rb") as f:
            cached = pickle.load(f)

    pending = [s for s in sids if s not in cached]
    if not pending:
        print(f": {EMB_CACHE} ({len(cached)} )")
        return cached

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_sleepfm_models(checkpoint, dev) if os.path.isfile(checkpoint) else None
    for i, sid in enumerate(pending, 1):
        x_dir = os.path.join(OUTPUT_DIR, "X", sid)
        emb, files, mode = extract_subject_embeddings(
            x_dir, checkpoint_path=checkpoint, models_tuple=models, dev=dev,
        )
        if emb is not None and len(emb) >= MIN_EPOCHS:
            cached[sid] = emb
            with open(EMB_CACHE, "wb") as f:
                pickle.dump(cached, f)
            if i % 20 == 0 or i == len(pending):
                print(f"   [{i}/{len(pending)}] {sid}: {len(files)} epochs ({mode})")
    return cached


def run(max_subjects: int = 0, skip_preprocess: bool = False) -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sids = list_apnea_patients()
    if max_subjects > 0:
        sids = sids[:max_subjects]
    print(f"APNEA  n={len(sids)}")

    if not skip_preprocess:
        print("[1/3]  patient_*.pt → SleepFM epoch ...")
        ok, fail = preprocess_all(skip_existing=True)
        sids = [s for s in sids if s in ok]
        print(f"       {len(ok)}，/ {len(fail)}")
    else:
        sids = [s for s in sids if os.path.isdir(os.path.join(OUTPUT_DIR, "X", s))]

    print("[2/3] SleepFM  ...")
    embeddings = _load_embeddings(sids, MODEL_CHECKPOINT)

    print("[3/3] UMAP  ...")
    outcomes = _load_outcomes()
    rows = []
    for i, sid in enumerate(sids, 1):
        emb = embeddings.get(sid)
        if emb is None:
            continue
        m = analyze_manifold_extended(emb)
        rows.append({"subject_id": sid, "n_epochs": len(emb), **m})
        if i % 25 == 0 or i == len(sids):
            print(f"      [{i}/{len(sids)}]  {len(rows)}")

    metrics = pd.DataFrame(rows)
    df = metrics.merge(outcomes, on="subject_id", how="left")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"metrics_apnea_{ts}.csv")
    df.to_csv(csv_path, index=False)
    meta = {
        "timestamp": ts,
        "n_subjects": len(df),
        "channels": "Flow/THO/ABD/ECG/SpO2 (no EEG, zero-padded to SleepFM 13ch)",
        "metrics": [f"{m}_{s}" for m in ALL_METRIC_NAMES for s in ("std", "mean")],
        "outcomes": ["ARI", "AHI"],
        "csv": csv_path,
    }
    json_path = os.path.join(OUTPUT_DIR, f"metrics_apnea_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n: {csv_path} (n={len(df)})")
    return df


def main():
    parser = argparse.ArgumentParser(description="APNEA ")
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--skip_preprocess", action="store_true")
    args = parser.parse_args()
    run(max_subjects=args.max_subjects, skip_preprocess=args.skip_preprocess)


if __name__ == "__main__":
    main()
