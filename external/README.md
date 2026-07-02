# External dependencies

This repository expects a few files that are **not** vendored here.

## SleepFM foundation model

Clone the official SleepFM codebase and download `best.pt`:

```bash
git clone <sleepfm-repo-url> third_party/sleepfm-codebase-main
# checkpoint -> third_party/sleepfm-codebase-main/sleepfm/checkpoint/best.pt
```

Set `SLEEPFM_ROOT` and `SLEEPFM_CHECKPOINT` in `.env` if needed.

## CPS cohort loader

CPS preprocessing imports `load.py` and `utils.py` from the CPS dataset repo:

```bash
export CPS_ROOT=/path/to/CPS
export PYTHONPATH=$CPS_ROOT:$PYTHONPATH
```

## DREAMT wristband filtering (optional)

`preprocess_dreamt_distill.py` may import bandpass helpers from DistillSleep. Install that project or provide equivalent utilities.

## Raw data layout

| Cohort  | Env var           | Notes |
|---------|-------------------|-------|
| CPS     | `CPS_DATA_ROOT`   | WFDB + questionnaire YAML |
| DREAMT  | `DREAMT_ROOT`     | `data_100Hz/`, `participant_info.csv` |
| Bidslab | `BIDSLAB_ROOT`    | watch CSV + `labels.mat` per night |
| APNEA   | `APNEA_DATA_ROOT` | `patient_*.pt`, `odi3_alignment.csv` |

See `env.example` for all paths.
