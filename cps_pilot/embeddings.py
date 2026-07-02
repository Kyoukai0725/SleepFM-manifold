"""Extract SleepFM 128-d embeddings from preprocessed epochs."""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config_cps import (
    CHANNEL_DATA_IDS,
    EMBEDDING_DIM,
    MODEL_CHECKPOINT,
    SLEEPFM_ROOT,
)

if SLEEPFM_ROOT not in sys.path:
    sys.path.insert(0, SLEEPFM_ROOT)
if os.path.join(SLEEPFM_ROOT, "model") not in sys.path:
    sys.path.insert(0, os.path.join(SLEEPFM_ROOT, "model"))

import models  # noqa: E402


def _strip_module_prefix(state_dict: dict) -> dict:

    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_sleepfm_models(checkpoint_path: str, device: torch.device):
    model_resp = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["Respiratory"]), stride=2, dilation=1)
    model_resp.fc = torch.nn.Linear(model_resp.fc.in_features, 512)
    model_sleep = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["Sleep_Stages"]), stride=2, dilation=1)
    model_sleep.fc = torch.nn.Linear(model_sleep.fc.in_features, 512)
    model_ekg = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["EKG"]), stride=2, dilation=1)
    model_ekg.fc = torch.nn.Linear(model_ekg.fc.in_features, 512)

    ckpt = torch.load(checkpoint_path, map_location=device)
    resp_key = "respiratory_state_dict" if "respiratory_state_dict" in ckpt else "resp_state_dict"
    sleep_key = "sleep_stages_state_dict" if "sleep_stages_state_dict" in ckpt else "sleep_state_dict"
    ekg_key = "ekg_state_dict"
    model_resp.load_state_dict(_strip_module_prefix(ckpt[resp_key]))
    model_sleep.load_state_dict(_strip_module_prefix(ckpt[sleep_key]))
    model_ekg.load_state_dict(_strip_module_prefix(ckpt[ekg_key]))
    for m in (model_resp, model_sleep, model_ekg):
        m.to(device)
        m.eval()
    return model_resp, model_sleep, model_ekg


def _stats_embedding(epoch: np.ndarray) -> np.ndarray:

    feats = []
    for ch in epoch:
        feats.extend([ch.mean(), ch.std(), ch.max(), ch.min()])
    return np.asarray(feats, dtype=np.float32)


def _stats_embeddings_batch(epochs: List[np.ndarray], dim: int = EMBEDDING_DIM) -> np.ndarray:
    raw = np.stack([_stats_embedding(e) for e in epochs], axis=0)
    raw = raw - raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-8
    raw = raw / std
    u, s, _ = np.linalg.svd(raw, full_matrices=False)
    k = min(dim, u.shape[1])
    emb = u[:, :k] * s[:k]
    if emb.shape[1] < dim:
        pad = np.zeros((emb.shape[0], dim - emb.shape[1]), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=1)
    return emb.astype(np.float32)


def _svd_reduce_128(combined: np.ndarray, dim: int = EMBEDDING_DIM) -> np.ndarray:

    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    centered = combined - combined.mean(0)
    if centered.shape[1] <= dim:
        return centered.astype(np.float32)
    rng = np.random.default_rng(0)
    for jitter in (0.0, 1e-6, 1e-4, 1e-2):
        try:
            arr = centered if jitter == 0 else centered + jitter * rng.standard_normal(centered.shape)
            u, s, _ = np.linalg.svd(arr.astype(np.float64), full_matrices=False)
            k = min(dim, len(s))
            out = u[:, :k] * s[:k]
            if out.shape[1] < dim:
                pad = np.zeros((out.shape[0], dim - out.shape[1]), dtype=np.float32)
                out = np.concatenate([out, pad], axis=1)
            return out.astype(np.float32)
        except np.linalg.LinAlgError:
            continue
    idx = np.linspace(0, centered.shape[1] - 1, dim).astype(int)
    return centered[:, idx].astype(np.float32)


@torch.no_grad()
def _sleepfm_embeddings_batch(
    epochs: List[np.ndarray],
    models_tuple,
    device: torch.device,
) -> np.ndarray:
    model_resp, model_sleep, model_ekg = models_tuple
    resp = torch.tensor(
        np.stack([e[CHANNEL_DATA_IDS["Respiratory"]] for e in epochs]),
        dtype=torch.float32,
        device=device,
    )
    sleep = torch.tensor(
        np.stack([e[CHANNEL_DATA_IDS["Sleep_Stages"]] for e in epochs]),
        dtype=torch.float32,
        device=device,
    )
    ekg = torch.tensor(
        np.stack([e[CHANNEL_DATA_IDS["EKG"]] for e in epochs]),
        dtype=torch.float32,
        device=device,
    )
    e_resp = torch.nn.functional.normalize(model_resp(resp), dim=1).cpu().numpy()
    e_sleep = torch.nn.functional.normalize(model_sleep(sleep), dim=1).cpu().numpy()
    e_ekg = torch.nn.functional.normalize(model_ekg(ekg), dim=1).cpu().numpy()
    combined = np.concatenate([e_resp, e_sleep, e_ekg], axis=1)
    return _svd_reduce_128(combined)


def extract_subject_embeddings(
    patient_x_dir: str,
    batch_size: int = 32,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    models_tuple=None,
    dev: Optional[torch.device] = None,
) -> Tuple[np.ndarray, List[str], str]:

    files = sorted(f for f in os.listdir(patient_x_dir) if f.endswith(".npy"))
    epochs = [np.load(os.path.join(patient_x_dir, f)) for f in files]
    if not epochs:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), [], "empty"

    use_ckpt = checkpoint_path and os.path.isfile(checkpoint_path)
    if use_ckpt:
        if dev is None:
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if models_tuple is None:
            models_tuple = _load_sleepfm_models(checkpoint_path, dev)
        chunks = []
        for i in range(0, len(epochs), batch_size):
            chunks.append(_sleepfm_embeddings_batch(epochs[i : i + batch_size], models_tuple, dev))
        emb = np.concatenate(chunks, axis=0)
        return emb, files, "sleepfm_checkpoint"

    emb = _stats_embeddings_batch(epochs, dim=EMBEDDING_DIM)
    return emb, files, "stats_fallback"
