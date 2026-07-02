"""Environment-based path helpers for local data and outputs."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT.parent))

SLEEPFM_ROOT = Path(
    os.environ.get(
        "SLEEPFM_ROOT",
        REPO_ROOT / "third_party" / "sleepfm-codebase-main" / "sleepfm",
    )
)
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "output"))

CPS_ROOT = Path(os.environ.get("CPS_ROOT", DATA_ROOT / "CPS"))
CPS_DATA = Path(os.environ.get("CPS_DATA_ROOT", CPS_ROOT / "data"))
DREAMT_ROOT = Path(os.environ.get("DREAMT_ROOT", DATA_ROOT / "watch" / "DREAMT"))
BIDSLAB_ROOT = Path(os.environ.get("BIDSLAB_ROOT", DATA_ROOT / "watch" / "Bidslab"))
APNEA_DATA_ROOT = Path(
    os.environ.get("APNEA_DATA_ROOT", DATA_ROOT / "APNEA_dataset" / "APNEA_EDF_processed_data")
)

CPS_OUTPUT = Path(os.environ.get("CPS_OUTPUT_DIR", OUTPUT_ROOT / "cps_pilot"))
CPS_PSQI_OUTPUT = Path(os.environ.get("CPS_PSQI_OUTPUT_DIR", OUTPUT_ROOT / "cps_psqi"))
DREAMT_OUTPUT = Path(os.environ.get("DREAMT_OUTPUT_DIR", OUTPUT_ROOT / "dreamt"))
BIDSLAB_OUTPUT = Path(os.environ.get("BIDSLAB_OUTPUT_DIR", OUTPUT_ROOT / "bidslab"))
APNEA_OUTPUT = Path(os.environ.get("APNEA_OUTPUT_DIR", OUTPUT_ROOT / "apnea"))

MODEL_CHECKPOINT = Path(
    os.environ.get("SLEEPFM_CHECKPOINT", SLEEPFM_ROOT / "checkpoint" / "best.pt")
)
