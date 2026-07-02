"""Bidslab multi-night Apple Watch cohort paths."""
import os
import sys

_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from env_paths import BIDSLAB_OUTPUT, BIDSLAB_ROOT  # noqa: E402

BIDSLAB_ROOT = str(BIDSLAB_ROOT)
OUTPUT_DIR = str(BIDSLAB_OUTPUT)
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
NIGHT_CACHE = os.path.join(CACHE_DIR, "nights")
UMAP_FRAME_DIR = os.path.join(CACHE_DIR, "umap_frames")
EPOCH_SEC = 30.0
TARGET_FS = 64.0
SAMPLES_PER_EPOCH = int(round(EPOCH_SEC * TARGET_FS))
EMBED_POOL = 64
SLEEP_STAGES = frozenset({1, 2, 3, 4})
UNKNOWN_STAGE = 5
OUTCOMES = ("TST_hours", "sleep_efficiency")
UMAP_N_NEIGHBORS = 30
UMAP_N_COMPONENTS = 3
UMAP_RANDOM_STATE = 0
