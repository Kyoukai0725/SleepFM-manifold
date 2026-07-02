"""Differentiable manifold metric proxies."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from config_cps import EPOCH_SEC, SLIDING_STEP_MIN, SLIDING_WINDOW_MIN

METRIC_NAMES = ("LCI", "SMI", "MFI", "VTI")
PHYSICAL_METRIC_NAMES = ("MTV", "MTC", "MTE", "MSV", "MRR")
ALL_METRIC_NAMES = METRIC_NAMES + PHYSICAL_METRIC_NAMES

SLIDING_WINDOW_EPOCHS = max(2, int(round(SLIDING_WINDOW_MIN * 60.0 / EPOCH_SEC)))
SLIDING_STEP_EPOCHS = max(1, int(round(SLIDING_STEP_MIN * 60.0 / EPOCH_SEC)))

TARGET_KEYS = [f"{m}_{s}" for m in ALL_METRIC_NAMES for s in ("std", "mean")]

PCA_RIDGE_EPS = 1e-6

_PROJ3: torch.Tensor | None = None


def pca_project_3d(x: torch.Tensor, ridge_eps: float = PCA_RIDGE_EPS) -> torch.Tensor:
    """Helper."""
    xc = x - x.mean(dim=0, keepdim=True)
    n, d = xc.shape
    if n < 3:
        return _random_project_3d(xc)

    cov = (xc.T @ xc) / max(n - 1, 1)
    cov = cov + ridge_eps * torch.eye(d, device=x.device, dtype=x.dtype)
    try:
        evals, evecs = torch.linalg.eigh(cov)
        k = min(3, evecs.shape[1])
        idx = torch.argsort(evals, descending=True)[:k]
        basis = evecs[:, idx]
        out = xc @ basis
        if out.shape[1] < 3:
            pad = torch.zeros(n, 3 - out.shape[1], device=x.device, dtype=x.dtype)
            out = torch.cat([out, pad], dim=1)
        return out
    except RuntimeError:
        return _random_project_3d(xc)


def _window_starts(n: int, win: int, step: int) -> List[int]:
    if n < win:
        return []
    return list(range(0, n - win + 1, step))


def _random_project_3d(xc: torch.Tensor) -> torch.Tensor:
    global _PROJ3
    if _PROJ3 is None or _PROJ3.device != xc.device or _PROJ3.shape[0] != xc.shape[1]:
        g = torch.Generator(device=xc.device)
        g.manual_seed(0)
        _PROJ3 = torch.randn(xc.shape[1], 3, device=xc.device, generator=g) / (xc.shape[1] ** 0.5)
    return xc @ _PROJ3


def log_euclidean_project(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    d = x.shape[1]
    cov = torch.cov(x.T) + eps * torch.eye(d, device=x.device, dtype=x.dtype)
    try:
        L = torch.linalg.cholesky(cov)
        return torch.linalg.solve(L, x.T).T
    except RuntimeError:
        centered = x - x.mean(dim=0, keepdim=True)
        return centered / (centered.std(dim=0, keepdim=True) + 1e-6)


def _lci(x: torch.Tensor) -> torch.Tensor:
    v1 = x[1:-1] - x[:-2]
    v2 = x[2:] - x[1:-1]
    return torch.linalg.norm(v2 - v1, dim=1).mean()


def _smi_proxy(x: torch.Tensor) -> torch.Tensor:
    """Helper."""
    if x.shape[0] < 8:
        return x.new_tensor(float("nan"))
    c = x.mean(dim=0, keepdim=True)
    d = torch.linalg.norm(x - c, dim=1)
    r = torch.quantile(d, 0.9) + 1e-6
    return (d < r).float().mean() / r


def _mfi_proxy(x: torch.Tensor) -> torch.Tensor:
    """Helper."""
    if x.shape[0] < 3:
        return x.new_tensor(float("nan"))
    step = torch.linalg.norm(x[1:] - x[:-1], dim=1)
    return step.std(unbiased=False)


def _vti(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Helper."""
    del k
    if x.shape[0] < 10:
        return x.new_tensor(float("nan"))
    vel = x[1:] - x[:-1]
    v0, v1 = vel[:-1], vel[1:]
    num = (v0 * v1).sum(dim=1)
    den = (torch.linalg.norm(v0, dim=1) * torch.linalg.norm(v1, dim=1)).clamp_min(1e-8)
    return (num / den).abs().mean()


def _mtv(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return x.new_tensor(float("nan"))
    return torch.linalg.norm(x[1:] - x[:-1], dim=1).mean()


def _mtc(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 3:
        return x.new_tensor(float("nan"))
    v = x[1:] - x[:-1]
    n = torch.linalg.norm(v, dim=1).clamp_min(1e-12)
    cos = (v[:-1] * v[1:]).sum(dim=1) / (n[:-1] * n[1:])
    return torch.acos(cos.clamp(-1.0, 1.0)).mean()


def _mte_proxy(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 4:
        return x.new_tensor(float("nan"))
    vel = torch.linalg.norm(x[1:] - x[:-1], dim=1)
    return vel.std(unbiased=False)


def _msv(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 4:
        return x.new_tensor(float("nan"))
    s = x.std(dim=0, unbiased=False).clamp_min(1e-8)
    return torch.log10(s.prod() ** (1.0 / s.numel()))


def _mrr(x: torch.Tensor, q: float = 0.10) -> torch.Tensor:
    del q
    n = x.shape[0]
    if n < 6:
        return x.new_tensor(float("nan"))
    dist = torch.cdist(x, x)
    dist = dist + torch.eye(n, device=x.device, dtype=x.dtype) * 1e6
    eps = dist.min(dim=1).values.mean() * 1.5 + 1e-8
    return (dist < eps).float().mean()


def _window_metrics(x_m: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "LCI": _lci(x_m),
        "SMI": _smi_proxy(x_m),
        "MFI": _mfi_proxy(x_m),
        "VTI": _vti(x_m),
        "MTV": _mtv(x_m),
        "MTC": _mtc(x_m),
        "MTE": _mte_proxy(x_m),
        "MSV": _msv(x_m),
        "MRR": _mrr(x_m),
    }


def differentiable_manifold_vector(
    embedding: torch.Tensor, ridge_eps: float = PCA_RIDGE_EPS
) -> torch.Tensor:
    """Helper."""
    nan_t = embedding.new_tensor(float("nan"))
    if embedding.shape[0] < SLIDING_WINDOW_EPOCHS:
        return torch.stack([nan_t] * len(TARGET_KEYS))

    x3 = pca_project_3d(embedding, ridge_eps=ridge_eps)
    rows: List[List[torch.Tensor]] = [[] for _ in ALL_METRIC_NAMES]

    for start in _window_starts(len(x3), SLIDING_WINDOW_EPOCHS, SLIDING_STEP_EPOCHS):
        chunk = x3[start : start + SLIDING_WINDOW_EPOCHS]
        xm = (chunk - chunk.mean(dim=0, keepdim=True)) / (chunk.std(dim=0, keepdim=True) + 1e-6)
        wm = _window_metrics(xm)
        for i, name in enumerate(ALL_METRIC_NAMES):
            rows[i].append(wm[name])

    parts: List[torch.Tensor] = []
    for i in range(len(ALL_METRIC_NAMES)):
        if len(rows[i]) < 2:
            parts.extend([nan_t, nan_t])
            continue
        vals = torch.stack(rows[i])
        if torch.isfinite(vals).sum() < 2:
            parts.extend([nan_t, nan_t])
            continue
        vf = vals[torch.isfinite(vals)]
        parts.append(vf.std(unbiased=False))
        parts.append(vf.mean())
    return torch.stack(parts)


def metric_vector_to_dict(vec: torch.Tensor) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for i, k in enumerate(TARGET_KEYS):
        v = vec[i]
        d[k] = float(v.item()) if torch.is_tensor(v) else float(v)
    return d
