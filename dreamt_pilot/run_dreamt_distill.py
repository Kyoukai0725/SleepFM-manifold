"""DREAMT wearable distillation entry point."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime

from distill_config import (
    DISTILLSLEEP_ROOT,
    FOLDS_DIR,
    OUTPUT_ROOT,
    TFRECORD_PAIRED,
    TFRECORD_PSG,
)

PYTHON = r"D:\miniconda\envs\psg_env311\python.exe"
FOLD = 0
INST = "dreamt"


def _run(cmd: list[str], cwd: str) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def _fold_dirs(kind: str, fold: int) -> tuple[str, str, str]:
    base = TFRECORD_PSG if kind == "psg" else TFRECORD_PAIRED
    return (
        os.path.join(base, f"fold{fold}", "train"),
        os.path.join(base, f"fold{fold}", "val"),
        os.path.join(base, f"fold{fold}", "test"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--max_subjects", type=int, default=0, help="0= 98")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_tfrecord", action="store_true")
    parser.add_argument("--teacher_epochs1", type=int, default=8)
    parser.add_argument("--teacher_epochs2", type=int, default=8)
    parser.add_argument("--student_epochs1", type=int, default=8)
    parser.add_argument("--student_epochs2", type=int, default=8)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pilot = os.path.dirname(os.path.abspath(__file__))

    if not args.skip_preprocess:
        cmd = [PYTHON, os.path.join(pilot, "preprocess_dreamt_distill.py")]
        if args.max_subjects > 0:
            cmd += ["--max_subjects", str(args.max_subjects)]
        _run(cmd, pilot)

    if not args.skip_tfrecord:
        _run([PYTHON, os.path.join(pilot, "build_dreamt_distill_tfrecords.py"), "--rebuild"], pilot)

    tr_psg, va_psg, te_psg = _fold_dirs("psg", args.fold)
    tr_pair, va_pair, te_pair = _fold_dirs("paired", args.fold)

    teacher_name = f"dreamt_teacher_f{args.fold}_{ts}"
    student_name = f"dreamt_student_kd_f{args.fold}_{ts}"

    common_teacher = [
        PYTHON, "train.py",
        "--institute", INST,
        "--fold", str(args.fold),
        "--num_channels", "3",
        "--num_epoch1", str(args.teacher_epochs1),
        "--num_epoch2", str(args.teacher_epochs2),
        "--batch_size", "32",
        "--stage1_model", "dwconv",
        "--stage2_model", "att_s2s_sm",
        "--window", "12",
        "--exp_name", teacher_name,
        "--loss", "wce",
        "--aug",
        "--paired_train_dir", tr_psg,
        "--paired_eval_dir", va_psg,
        "--paired_test_dir", te_psg,
    ]
    _run(common_teacher, DISTILLSLEEP_ROOT)

    common_student = [
        PYTHON, "kd_train.py",
        "--institute", INST,
        "--fold", str(args.fold),
        "--paired_data",
        "--dim_psg", "6000", "3",
        "--dim_watch", "1920", "3",
        "--num_channels", "3",
        "--num_epoch1", str(args.student_epochs1),
        "--num_epoch2", str(args.student_epochs2),
        "--batch_size", "32",
        "--stage1_model", "dwconv",
        "--stage2_model", "att_s2s_sm",
        "--window", "12",
        "--teacher_ckpt", teacher_name,
        "--exp_name", student_name,
        "--loss", "wce",
        "--s1_feature_kd_method", "OFD",
        "--s1_logit_kd_method", "DIST",
        "--s2_feature_kd_method", "MSE",
        "--s2_logit_kd_method", "DIST",
        "--paired_train_dir", tr_pair,
        "--paired_eval_dir", va_pair,
        "--paired_test_dir", te_pair,
    ]
    _run(common_student, DISTILLSLEEP_ROOT)

    summary = {
        "fold": args.fold,
        "teacher_ckpt": teacher_name,
        "student_ckpt": student_name,
        "distillsleep_root": DISTILLSLEEP_ROOT,
        "output_root": OUTPUT_ROOT,
    }
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_path = os.path.join(OUTPUT_ROOT, f"distill_run_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n。: {summary_path}")


if __name__ == "__main__":
    main()
