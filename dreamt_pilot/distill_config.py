"""Distillation hyperparameters."""
import os

DREAMT_ROOT = r"E:\psg_dataset\watch\DREAMT"
DATA_100HZ = os.path.join(DREAMT_ROOT, "data_100Hz")
DISTILLSLEEP_ROOT = r"E:\psg_dataset\watch\DistillSleep-main"

OUTPUT_ROOT = r"E:\psg_dataset\SleepFM\output\dreamt\distill"
NPY_PSG = os.path.join(OUTPUT_ROOT, "npy", "psg")
NPY_WATCH = os.path.join(OUTPUT_ROOT, "npy", "watch")
NPY_ANN = os.path.join(OUTPUT_ROOT, "npy", "ann")
TFRECORD_PSG = os.path.join(OUTPUT_ROOT, "tfrecord_psg")
TFRECORD_PAIRED = os.path.join(OUTPUT_ROOT, "tfrecord_paired")
FOLDS_DIR = os.path.join(OUTPUT_ROOT, "folds")

SOURCE_FS = 100.0
PSG_TARGET_FS = 200
WATCH_TARGET_FS = 64
EPOCH_SEC = 30
SAMPLES_SRC = int(EPOCH_SEC * SOURCE_FS)
SAMPLES_PSG = PSG_TARGET_FS * EPOCH_SEC  # 6000
SAMPLES_WATCH = WATCH_TARGET_FS * EPOCH_SEC  # 1920

PSG_COLS = ["C4-M1", "E1", "CHIN"]
WATCH_COLS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "EDA"]

STAGE_TO_LABEL = {
    "W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4,
    "P": 0, "Missing": 0,
}

N_FOLDS = 5
SEED = 2481757
