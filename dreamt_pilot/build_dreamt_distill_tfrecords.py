"""Build TFRecords and CV folds."""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tqdm import tqdm

from distill_config import (
    FOLDS_DIR,
    NPY_ANN,
    NPY_PSG,
    NPY_WATCH,
    N_FOLDS,
    SAMPLES_PSG,
    SAMPLES_WATCH,
    SEED,
    TFRECORD_PAIRED,
    TFRECORD_PSG,
)


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _write_psg_only_records(ids: list[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for sid in tqdm(ids, desc=f"PSG → {out_dir}"):
        out_path = os.path.join(out_dir, f"{sid}.records")
        if os.path.isfile(out_path):
            continue
        psg = np.load(os.path.join(NPY_PSG, f"{sid}.npy"))
        ann = np.load(os.path.join(NPY_ANN, f"{sid}.npy"))
        n = min(len(psg), len(ann))
        with tf.io.TFRecordWriter(out_path) as writer:
            for i in range(n):
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "signal": _float_feature(list(psg[i].reshape(-1))),
                    "ann": _int64_feature([int(ann[i])]),
                    "epoch_idx": _int64_feature([int(i)]),
                }))
                writer.write(ex.SerializeToString())


def _write_paired_records(ids: list[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for sid in tqdm(ids, desc=f"Paired → {out_dir}"):
        out_path = os.path.join(out_dir, f"{sid}.records")
        if os.path.isfile(out_path):
            continue
        psg = np.load(os.path.join(NPY_PSG, f"{sid}.npy"))
        watch = np.load(os.path.join(NPY_WATCH, f"{sid}.npy"))
        ann = np.load(os.path.join(NPY_ANN, f"{sid}.npy"))
        n = min(len(psg), len(watch), len(ann))
        with tf.io.TFRecordWriter(out_path) as writer:
            for i in range(n):
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "signal_psg": _float_feature(list(psg[i].reshape(-1))),
                    "signal_watch": _float_feature(list(watch[i].reshape(-1))),
                    "ann": _int64_feature([int(ann[i])]),
                    "epoch_idx": _int64_feature([int(i)]),
                }))
                writer.write(ex.SerializeToString())


def _make_folds(all_ids: list[str]) -> dict:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = {}
    ids = np.array(all_ids)
    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(ids)):
        test_ids = ids[te_idx].tolist()
        trainval_ids = ids[tr_idx].tolist()
        n_val = max(1, len(trainval_ids) // 5)
        folds[fold_id] = {
            "train": trainval_ids[n_val:],
            "val": trainval_ids[:n_val],
            "test": test_ids,
        }
    return folds


def build_fold(folds: dict, fold_id: int) -> None:
    split_map = folds[fold_id]
    for split, sids in split_map.items():
        _write_psg_only_records(sids, os.path.join(TFRECORD_PSG, f"fold{fold_id}", split))
        _write_paired_records(sids, os.path.join(TFRECORD_PAIRED, f"fold{fold_id}", split))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1, help="-1=，")
    args = parser.parse_args()

    all_ids = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in os.listdir(NPY_PSG)
        if p.endswith(".npy")
    )
    if not all_ids:
        raise FileNotFoundError(f"")

    os.makedirs(FOLDS_DIR, exist_ok=True)
    fold_pkl = os.path.join(FOLDS_DIR, "dreamt_folds.pkl")
    if os.path.isfile(fold_pkl):
        with open(fold_pkl, "rb") as f:
            folds = pickle.load(f)
    else:
        folds = _make_folds(all_ids)
        with open(fold_pkl, "wb") as f:
            pickle.dump(folds, f)
        with open(os.path.join(FOLDS_DIR, "dreamt_folds.json"), "w", encoding="utf-8") as f:
            json.dump(folds, f, indent=2)

    targets = range(N_FOLDS) if args.fold < 0 else [args.fold]
    for fid in targets:
        print(f" fold {fid}: train={len(folds[fid]['train'])} val={len(folds[fid]['val'])} test={len(folds[fid]['test'])}")
        build_fold(folds, fid)
    print(f"。PSG epoch  ({SAMPLES_PSG},3)，Watch ({SAMPLES_WATCH},3)")


if __name__ == "__main__":
    main()
