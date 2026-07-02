# SleepFM Manifold Dynamics

Preprocessing and analysis code for **UMAP-3D sleep manifold dynamics** across four cohorts.

| Cohort | Signal | Entry script |
|--------|--------|--------------|
| CPS | Lab PSG → SleepFM 128d | `cps_pilot/run_pilot.py` |
| DREAMT | PSG teacher + E4 wristband student | `dreamt_pilot/run_dreamt_manifold.py`, `train_manifold_distill.py` |
| Bidslab | Apple Watch multi-night | `bidslab_pilot/run_bidslab_pipeline.py` |
| APNEA | 5-channel Level-3 PSG | `apnea_pilot/run_apnea_manifold.py` |

Cross-cohort: `run_super_metrics.py`, `run_manifold_fa.py`.

**Not included:** figures/tables, cached outputs, checkpoints, or raw data (see `.gitignore`).

## Pipeline

```
Raw signals → 30 s epochs → SleepFM 128d embeddings
           → UMAP 3D → 30 min sliding windows
           → 10 metrics (LCI/SMI/MFI/VTI + MTV/MTC/MTE/MSV/MRR/LZC)
           → nightly *_std / *_mean → clinical correlations
```

DREAMT distillation trains a wristband student to match 18 PSG teacher scalars (9 metrics × std/mean).

## Setup

```bash
conda create -n sleepfm_manifold python=3.11 -y
conda activate sleepfm_manifold
pip install -r requirements.txt
cp env.example .env   # edit paths
```

Install SleepFM and external loaders — see [external/README.md](external/README.md).

## Quick start

```bash
python cps_pilot/run_pilot.py
python dreamt_pilot/run_dreamt_manifold.py
python dreamt_pilot/train_manifold_distill.py --fold -1
python bidslab_pilot/run_bidslab_pipeline.py
python apnea_pilot/run_apnea_manifold.py
python run_super_metrics.py --cohort all
python run_super_metrics.py --cohort all --with-lzc   # optional MFI+LZC
```

Outputs go to `output/` (gitignored).

## Layout

```
sleepfm-manifold/
├── lib/              env_paths.py, stats_utils.py
├── cps_pilot/        shared core + CPS pipeline
├── dreamt_pilot/     PSG + wristband distillation
├── bidslab_pilot/    Apple Watch multi-night
├── apnea_pilot/      Level-3 validation
├── external/         third-party notes
├── run_super_metrics.py
└── run_manifold_fa.py
```

## Super-metrics

- **Trajectory_Activity** = mean(z): LCI + MTV + MTC
- **Stability** = mean(z): SMI + VTI
- **Trajectory_Irregularity** = z(MFI) [+ LZC with `--with-lzc`]

## Rebuild this bundle

From the full research repo:

```bash
python tools/build_github_release.py
python tools/finalize_github_release.py
python tools/sweep_chinese_release.py
```

## License

Research code for reproducibility. SleepFM weights and cohort data are subject to their original licenses.
