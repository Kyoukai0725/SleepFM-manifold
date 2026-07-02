"""APNEA cohort: 5-channel Level-3 PSG pipeline."""
import os
import sys

_CPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cps_pilot")
if _CPS not in sys.path:
    sys.path.insert(0, _CPS)
_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from env_paths import APNEA_DATA_ROOT, APNEA_OUTPUT  # noqa: E402
from config_cps import ALL_CHANNELS, EPOCH_SEC, MODEL_CHECKPOINT, TARGET_FS  # noqa: E402

APNEA_DATA_ROOT = str(APNEA_DATA_ROOT)
ODI_ALIGNMENT_CSV = os.path.join(APNEA_DATA_ROOT, "odi3_alignment.csv")
OUTPUT_DIR = str(APNEA_OUTPUT)
APNEA_CHANNEL_NAMES = ["Flow Patient", "Effort THO", "Effort ABD", "ECG I", "SpO2"]
APNEA_SOURCE_FS = 100
APNEA_SAMPLES_PER_EPOCH = 3000
APNEA_TO_SLEEPFM = {
    0: ALL_CHANNELS.index("AIRFLOW"), 1: ALL_CHANNELS.index("CHEST"),
    2: ALL_CHANNELS.index("ABD"), 3: ALL_CHANNELS.index("ECG"), 4: ALL_CHANNELS.index("SaO2"),
}
MIN_EPOCHS = 60
