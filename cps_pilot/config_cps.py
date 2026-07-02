"""CPS cohort paths and SleepFM channel mapping."""
import os
import sys

_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from env_paths import CPS_DATA, CPS_OUTPUT, MODEL_CHECKPOINT, SLEEPFM_ROOT  # noqa: E402

CPS_DATA_ROOT = str(CPS_DATA)
OUTPUT_DIR = str(CPS_OUTPUT)
SLEEPFM_ROOT = str(SLEEPFM_ROOT)
MODEL_CHECKPOINT = str(MODEL_CHECKPOINT)

ALL_CHANNELS = [
    "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1",
    "E1-M2", "Chin1-Chin2", "ABD", "CHEST", "AIRFLOW", "SaO2", "ECG",
]

CPS_CHANNEL_CANDIDATES = {
    "F3-M2": ["C3:A2", "C3"], "F4-M1": ["F4:A1", "F4"], "C3-M2": ["C3:A2", "C3"],
    "C4-M1": ["C4:A1", "C4"], "O1-M2": ["O2:A1", "A1:O2", "O2"], "O2-M1": ["O2:A1", "A1:O2", "O2"],
    "E1-M2": ["EOGl", "EOGl:A1", "EOGl:A2"], "Chin1-Chin2": ["EMG", "EMG+"],
    "ABD": ["RIP.Abdom"], "CHEST": ["RIP.Thrx"], "AIRFLOW": ["Druck Flow", "Flow Th"],
    "SaO2": ["SPO2"], "ECG": ["ECG 2"],
}
CPS_CHANNEL_MAP = {k: v[0] for k, v in CPS_CHANNEL_CANDIDATES.items()}
CHANNEL_DATA = {"Respiratory": ["CHEST", "SaO2", "ABD"], "Sleep_Stages": ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2"], "EKG": ["ECG"]}
CHANNEL_DATA_IDS = {k: [ALL_CHANNELS.index(c) for c in v] for k, v in CHANNEL_DATA.items()}
CPS_STAGE_MAP = {"Wach": "Wake", "N1": "Stage 1", "N2": "Stage 2", "N3": "Stage 3", "Rem": "REM"}
TARGET_FS = 256
EPOCH_SEC = 30.0
BANDPASS = (0.5, 30.0)
EMBEDDING_DIM = 128
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 30
SLIDING_WINDOW_MIN = 30.0
SLIDING_STEP_MIN = 15.0
