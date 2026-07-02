"""DREAMT cohort paths and PSG channel mapping."""
import os
import sys

_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from env_paths import DREAMT_OUTPUT, DREAMT_ROOT, MODEL_CHECKPOINT, SLEEPFM_ROOT  # noqa: E402

DREAMT_ROOT = str(DREAMT_ROOT)
DATA_100HZ = os.path.join(DREAMT_ROOT, "data_100Hz")
PARTICIPANT_CSV = os.path.join(DREAMT_ROOT, "participant_info.csv")
OUTPUT_DIR = str(DREAMT_OUTPUT)
DISTILL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "distill")
TEACHER_EMB_CACHE = os.path.join(OUTPUT_DIR, "cache", "embeddings_128d.pkl")
SLEEPFM_ROOT = str(SLEEPFM_ROOT)
MODEL_CHECKPOINT = str(MODEL_CHECKPOINT)
SOURCE_FS = 100.0
TARGET_FS = 256.0
EPOCH_SEC = 30.0
BANDPASS = (0.5, 30.0)
ALL_CHANNELS = [
    "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1",
    "E1-M2", "Chin1-Chin2", "ABD", "CHEST", "AIRFLOW", "SaO2", "ECG",
]
DREAMT_CHANNEL_CANDIDATES = {
    "F3-M2": ["Fp1-O2", "T3 - CZ"], "F4-M1": ["F4-M1"], "C3-M2": ["T3 - CZ", "CZ - T4"],
    "C4-M1": ["C4-M1"], "O1-M2": ["Fp1-O2", "O2-M1"], "O2-M1": ["O2-M1"],
    "E1-M2": ["E1", "E2"], "Chin1-Chin2": ["CHIN"], "ABD": ["ABDOMEN"], "CHEST": ["THORAX"],
    "AIRFLOW": ["FLOW"], "SaO2": ["SAO2"], "ECG": ["ECG"],
}
PSG_COLS = sorted({c for v in DREAMT_CHANNEL_CANDIDATES.values() for c in v})
OUTCOMES = ("AHI", "ARI", "TST_hours", "sleep_efficiency")
