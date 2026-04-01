"""evaluation/toolbox_adapter.py

Bridge layer between rPPG-Toolbox (ubicomplab/rPPG-Toolbox) and SoFake.

Handles
-------
* sys.path injection so rPPG-Toolbox modules are importable
* DiffNormalized preprocessing  (required by all Toolbox neural models)
* Unified model-loading for the 9 supported architectures
* Graceful fallback to the original SoFake PhysNet(S=2) when the Toolbox
  is not installed

Setup
-----
  git clone https://github.com/nia1003/rPPG-Toolbox.git ../rPPG-Toolbox
  # -or- set environment variable:
  export RPPG_TOOLBOX_ROOT=/path/to/rPPG-Toolbox
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1.  Toolbox discovery
# ---------------------------------------------------------------------------

def _find_toolbox() -> Path | None:
    env = os.environ.get("RPPG_TOOLBOX_ROOT")
    if env and (Path(env) / "neural_methods").exists():
        return Path(env)
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "rPPG-Toolbox",
        Path(__file__).resolve().parent.parent / "rPPG-Toolbox",
    ]
    for c in candidates:
        if (c / "neural_methods").exists():
            return c
    return None


TOOLBOX_ROOT      = _find_toolbox()
TOOLBOX_AVAILABLE = TOOLBOX_ROOT is not None

if TOOLBOX_AVAILABLE:
    sys.path.insert(0, str(TOOLBOX_ROOT))


# ---------------------------------------------------------------------------
# 2.  Preprocessing
# ---------------------------------------------------------------------------

def diff_normalize(frames: np.ndarray) -> np.ndarray:
    """DiffNormalized transform required by rPPG-Toolbox neural models.

    Parameters
    ----------
    frames : (T, H, W, 3)  float32  RGB  values in [0, 255]

    Returns
    -------
    (T, H, W, 3)  float32  DiffNormalized

    Formula
    -------
        d[t] = (f[t+1] - f[t]) / (f[t+1] + f[t] + 1e-7)
        output = d / std(d)
    """
    T   = frames.shape[0]
    out = np.zeros_like(frames, dtype=np.float32)
    for t in range(T - 1):
        out[t] = (frames[t + 1] - frames[t]) / (frames[t + 1] + frames[t] + 1e-7)
    out[T - 1] = out[T - 2]           # replicate last diff
    std = float(np.std(out))
    if std > 1e-8:
        out /= std
    return out


def bgr_to_rgb_float(frames: np.ndarray) -> np.ndarray:
    """(T,H,W,3) uint8 BGR  ->  (T,H,W,3) float32 RGB in [0,255]."""
    return frames[:, :, :, ::-1].astype(np.float32)


def preprocess(frames_bgr: np.ndarray,
               size: int = 72,
               data_type: str = "DiffNormalized") -> np.ndarray:
    """Full preprocessing pipeline: resize  BGR->RGB  then normalize.

    Parameters
    ----------
    frames_bgr : (T, H, W, 3) uint8 BGR
    size       : target spatial dimension (72 for Toolbox standard)
    data_type  : "DiffNormalized"  |  "Raw"

    Returns
    -------
    (T, size, size, 3) float32
    """
    # Resize
    resized = np.stack(
        [cv2.resize(f, (size, size)) for f in frames_bgr], axis=0
    )  # (T, size, size, 3) uint8

    rgb = bgr_to_rgb_float(resized)   # float32, [0,255]

    if data_type == "DiffNormalized":
        return diff_normalize(rgb)
    else:                              # Raw: normalize to [0,1]
        out = rgb / 255.0
        m   = out.mean(axis=(0, 1, 2), keepdims=True)
        s   = out.std(axis=(0, 1, 2),  keepdims=True) + 1e-8
        return (out - m) / s


def to_tensor(frames: np.ndarray, device) -> torch.Tensor:
    """(T,H,W,3) float32  ->  (1,3,T,H,W) tensor."""
    arr = frames.transpose(3, 0, 1, 2)          # (3,T,H,W)
    return torch.from_numpy(arr.copy()).unsqueeze(0).float().to(device)


# ---------------------------------------------------------------------------
# 3.  Model registry
# ---------------------------------------------------------------------------

# (module_path_relative_to_toolbox_root, class_name)
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "PhysNet":       ("neural_methods.model.PhysNet",
                      "PhysNet_padding_Encoder_Decoder_MAX"),
    "DeepPhys":      ("neural_methods.model.DeepPhys",      "DeepPhys"),
    "EfficientPhys": ("neural_methods.model.EfficientPhys", "EfficientPhys"),
    "TS-CAN":        ("neural_methods.model.TS_CAN",        "TSCAN"),
    "PhysFormer":    ("neural_methods.model.PhysFormer.PhysFormer_ED",
                      "PhysFormer_ED"),
    "PhysMamba":     ("neural_methods.model.PhysMamba",     "PhysMamba"),
    "RhythmFormer":  ("neural_methods.model.RhythmFormer.RhythmFormer",
                      "RhythmFormer"),
    "FactorizePhys": ("neural_methods.model.FactorizePhys.FactorizePhys",
                      "FactorizePhys"),
    "iBVPNet":       ("neural_methods.model.iBVPNet",       "iBVPNet"),
}

# Data-type each model expects
MODEL_DATA_TYPE: dict[str, str] = {
    "PhysNet":       "DiffNormalized",
    "DeepPhys":      "DiffNormalized",
    "EfficientPhys": "DiffNormalized",
    "TS-CAN":        "DiffNormalized",
    "PhysFormer":    "DiffNormalized",
    "PhysMamba":     "DiffNormalized",
    "RhythmFormer":  "DiffNormalized",
    "FactorizePhys": "DiffNormalized",
    "iBVPNet":       "Raw",
}

# Clip length each model was trained with
MODEL_CLIP_LEN: dict[str, int] = {
    "PhysNet":       128,
    "DeepPhys":      128,
    "EfficientPhys": 128,
    "TS-CAN":        128,
    "PhysFormer":    128,
    "PhysMamba":     128,
    "RhythmFormer":  128,
    "FactorizePhys": 160,
    "iBVPNet":       128,
}


def available_models() -> list[str]:
    return list(_MODEL_REGISTRY.keys())


def load_toolbox_model(name: str,
                       ckpt_path: str,
                       device,
                       frames: int | None = None) -> nn.Module:
    """Instantiate and load a rPPG-Toolbox model from a checkpoint.

    Parameters
    ----------
    name      : one of available_models()
    ckpt_path : path to .pth checkpoint
    device    : torch device
    frames    : temporal clip length (defaults to MODEL_CLIP_LEN[name])
    """
    if not TOOLBOX_AVAILABLE:
        raise ImportError(
            "rPPG-Toolbox not found.\n"
            "Clone it:  git clone https://github.com/nia1003/rPPG-Toolbox.git "
            "../rPPG-Toolbox\n"
            "Then re-run."
        )

    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {available_models()}"
        )

    if frames is None:
        frames = MODEL_CLIP_LEN.get(name, 128)

    mod_path, cls_name = _MODEL_REGISTRY[name]
    mod   = importlib.import_module(mod_path)
    Cls   = getattr(mod, cls_name)

    try:
        model = Cls(frames=frames).to(device)
    except TypeError:
        model = Cls().to(device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip 'module.' prefix from DataParallel checkpoints
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_inference(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Run a Toolbox model on a (1,3,T,H,W) tensor.

    Returns the rPPG waveform as a (T,) numpy array.

    rPPG-Toolbox models return either:
      - A tuple  (rppg, *visual_features)
      - A single tensor  rppg
    """
    with torch.no_grad():
        out = model(tensor)
    rppg = out[0] if isinstance(out, (tuple, list)) else out
    return rppg.squeeze().cpu().numpy()          # (T,)


# ---------------------------------------------------------------------------
# 4.  SoFake-legacy fallback  (PhysNet S=2)
# ---------------------------------------------------------------------------

class _SoFakeFallback(nn.Module):
    """Thin wrapper that loads the original SoFake PhysNet(S=2) and exposes
    the same (rppg,) output interface as rPPG-Toolbox models."""

    def __init__(self, ckpt_path: str, device):
        super().__init__()
        _here = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(_here))
        from model.physnet_model import PhysNet
        self.physnet = PhysNet(S=2).to(device)
        state = torch.load(ckpt_path, map_location=device)
        self.physnet.load_state_dict(state)
        self.physnet.eval()

    def forward(self, x):
        # SoFake PhysNet returns (B, 5, T); average spatial channels -> (B, T)
        out = self.physnet(x)          # (B, 5, T)
        return (out.mean(dim=1),)      # tuple for consistency with Toolbox API


def load_sofake_fallback(ckpt_path: str, device) -> nn.Module:
    return _SoFakeFallback(ckpt_path, device)
