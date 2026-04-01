"""evaluation/fusion_evaluator.py  (rPPG-Toolbox edition)

Pipeline
--------
  1. Read video frames, detect & crop face region
  2. Apply DiffNormalized preprocessing (rPPG-Toolbox standard)
  3. Run rPPG-Toolbox neural model (default: PhysNet) on 128-frame clips
  4. Compute per-window scores
     * rppg_score : cardiac-band SNR  (peak/noise-floor, tanh-normalised)
                    Higher -> stronger cardiac signal -> likely real
     * fft_score  : cardiac-band energy fraction from a 512-sample rolling
                    accumulator; rfft+argmax, stable across all window counts
  5. Return list of {window, rppg_score, fft_score} dicts

Backward compatibility
----------------------
  If rPPG-Toolbox is not installed, falls back to the original SoFake
  PhysNet(S=2) with the existing model_weights.pt checkpoint.

Usage
-----
  python3 evaluation/fusion_evaluator.py \\
      --video  path/to/video.mp4 \\
      --weights path/to/PURE_PhysNet_DiffNormalized.pth \\
      --model  PhysNet
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.toolbox_adapter import (
    TOOLBOX_AVAILABLE,
    MODEL_CLIP_LEN,
    MODEL_DATA_TYPE,
    diff_normalize,
    preprocess,
    to_tensor,
    load_toolbox_model,
    load_sofake_fallback,
    run_inference,
)
from utils.utils_sig import butter_bandpass

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class FusionEvaluator:
    """
    rPPG-Toolbox powered deepfake evaluator.

    Changes vs. previous SoFake versions
    -------------------------------------
    | Aspect              | Before          | Now (rPPG-Toolbox)           |
    |---------------------|-----------------|------------------------------|
    | Model               | PhysNet S=2     | Any Toolbox model (9 choices)|
    | Preprocessing       | raw RGB /255    | DiffNormalized               |
    | Spatial resolution  | 128×128         | 72×72  (Toolbox standard)    |
    | Clip length         | 30 frames       | 128 frames  (4.25 s @ 30fps) |
    | Output signal       | (B,5,T) spatial | (B,T) rPPG waveform          |
    | rppg_score basis    | band-power ratio| cardiac-band SNR             |
    | FFT resolution      | ~1 Hz / bin     | ~0.23 Hz / bin (much better) |
    """

    # rPPG-Toolbox standard parameters
    _FACE_SZ   = 72     # px — Toolbox trained at 72×72
    _STRIDE_R  = 0.5    # clip stride as fraction of clip length (50 % overlap)
    _FFT_ACCUM = 512    # rolling accumulator length (samples)
    _FFT_MIN   = 128    # minimum samples before attempting FFT

    def __init__(self,
                 video_path : str,
                 weight_path: str,
                 model_name : str = "PhysNet"):
        """
        Parameters
        ----------
        video_path  : path to input video
        weight_path : .pth checkpoint (rPPG-Toolbox) or model_weights.pt (SoFake)
        model_name  : rPPG-Toolbox model key  (ignored when Toolbox unavailable)
        """
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_path = Path(video_path).resolve()
        self.model_name = model_name

        if TOOLBOX_AVAILABLE:
            self._clip_len  = MODEL_CLIP_LEN.get(model_name, 128)
            self._data_type = MODEL_DATA_TYPE.get(model_name, "DiffNormalized")
            self._model     = load_toolbox_model(
                model_name, weight_path, self.device, frames=self._clip_len
            )
            log.info(f"[rPPG-Toolbox] {model_name} loaded from {weight_path} "
                     f"on {self.device}  (clip={self._clip_len}, "
                     f"preproc={self._data_type})")
        else:
            log.warning(
                "rPPG-Toolbox not found — falling back to SoFake PhysNet(S=2). "
                "To use the Toolbox, run:  bash scripts/setup_rppg_toolbox.sh"
            )
            self._clip_len  = 128
            self._data_type = "Raw"
            self._model     = load_sofake_fallback(weight_path, self.device)
            log.info(f"SoFake PhysNet(S=2) loaded from {weight_path}")

        self._stride = max(1, int(self._clip_len * self._STRIDE_R))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _znorm(x: np.ndarray) -> np.ndarray:
        s = x.std()
        return (x - x.mean()) / (s if s > 1e-8 else 1.0)

    # ---- Score 1: cardiac-band SNR on a single clip signal -----------

    @staticmethod
    def _rppg_snr_score(signal: np.ndarray, fps: float) -> float:
        """peak cardiac-band FFT power / noise-floor, tanh-normalised to [0,1].

        Higher score -> stronger cardiac signal -> more likely real face.
        """
        try:
            sig      = FusionEvaluator._znorm(signal)
            windowed = sig * np.hanning(len(sig))
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs    = np.fft.rfftfreq(len(windowed), d=1.0 / fps)

            cmask = (freqs >= 0.6) & (freqs <= 4.0)
            nmask = (~cmask) & (freqs > 0.0)
            if not cmask.any() or not nmask.any():
                return 0.0

            peak  = float(np.max(spectrum[cmask]))
            noise = float(np.mean(spectrum[nmask]))
            snr   = peak / (noise + 1e-8)
            return float(np.tanh(snr / 3.0))          # map (0,∞) -> (0,1)
        except Exception:
            return 0.0

    # ---- Score 2: cardiac-band energy from rolling accumulator -------

    def _fft_score(self,
                   accum: np.ndarray,
                   fps  : float) -> tuple[float, float]:
        """Returns (score ∈ [0,1], dominant_hr_bpm)."""
        if len(accum) < self._FFT_MIN:
            return 0.0, 0.0
        try:
            sig      = accum - np.linspace(accum[0], accum[-1], len(accum))
            filtered = butter_bandpass(sig, lowcut=0.6, highcut=4.0,
                                       fs=fps)
            windowed = filtered * np.hanning(len(filtered))
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs    = np.fft.rfftfreq(len(windowed), d=1.0 / fps)

            cmask = (freqs >= 0.6) & (freqs <= 4.0)
            if not cmask.any():
                return 0.0, 0.0

            cs       = spectrum.copy(); cs[~cmask] = 0.0
            hr_bpm   = float(freqs[np.argmax(cs)] * 60.0)
            score    = float(np.clip(
                np.sum(spectrum[cmask]) / (np.sum(spectrum) + 1e-8),
                0.0, 1.0
            ))
            return score, hr_bpm
        except Exception:
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> list[dict]:
        """Process the video and return per-window score dicts.

        Returns
        -------
        list of {window: int, rppg_score: float, fft_score: float}
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_buf = deque(maxlen=self._clip_len)
        rppg_acc  = deque(maxlen=self._FFT_ACCUM)
        results   : list[dict] = []
        frame_cnt = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buf.append(frame)     # store raw BGR uint8
            frame_cnt += 1

            if (len(frame_buf) == self._clip_len
                    and frame_cnt % self._stride == 0):

                # ----- Stage 1: preprocessing + inference --------
                frames_np = np.stack(list(frame_buf), axis=0)   # (T,H,W,3)
                proc      = preprocess(
                    frames_np,
                    size      = self._FACE_SZ,
                    data_type = self._data_type,
                )                                               # (T,72,72,3)
                clip      = to_tensor(proc, self.device)       # (1,3,T,72,72)
                signal    = run_inference(self._model, clip)   # (T,)

                # ----- Stage 2: scores ---------------------------
                r_score = self._rppg_snr_score(signal, fps)

                # Append newest stride samples to accumulator
                chunk = self._znorm(signal[-self._stride:])
                rppg_acc.extend(chunk.tolist())
                f_score, hr_bpm = self._fft_score(
                    np.array(rppg_acc), fps
                )

                results.append({
                    "window":     frame_cnt,
                    "rppg_score": round(r_score, 4),
                    "fft_score":  round(f_score,  4),
                })
                log.info(
                    f"[window={frame_cnt:>5d}]  "
                    f"snr={r_score:.4f}  "
                    f"fft={f_score:.4f}  "
                    f"HR={hr_bpm:.1f}bpm  "
                    f"(accum={len(rppg_acc)})"
                )

        cap.release()
        return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="rPPG-Toolbox fusion evaluator"
    )
    parser.add_argument("--video",
        default="../datasets/FaceForensics/original_sequences/"
                "youtube/c23/videos/183.mp4")
    parser.add_argument("--weights",
        default="../rPPG-Toolbox/final_model_release/"
                "PURE_PhysNet_DiffNormalized.pth")
    parser.add_argument("--model",   default="PhysNet",
        help=f"rPPG-Toolbox model name")
    parser.add_argument("--output",  default="results/183_toolbox.csv")
    args = parser.parse_args()

    ev   = FusionEvaluator(args.video, args.weights, args.model)
    data = ev.evaluate()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window", "rppg_score", "fft_score"])
        w.writeheader()
        w.writerows(data)
    log.info(f"Saved {args.output}  ({len(data)} windows)")
