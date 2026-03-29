import csv
import logging
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from model.physnet_model import PhysNet
from utils.utils_sig import butter_bandpass

logging.basicConfig(level=logging.INFO, format="%(message)s")


class FusionEvaluator:
    """
    Two-stage rPPG deepfake evaluator  (v2 – spatial-consistency scorer).

    Stage 1 – PhysNet (3D CNN):
        Input  : (1, 3, T, 128, 128)  RGB, [0,1]
        Output : (1, N, T)  ST-rPPG block, N = S*S+1 = 5  (S=2)
                  channels 0-3 → spatial grid positions
                  channel  4   → spatial average

    Stage 2 – Three complementary scores:

        rppg_score  (PRIMARY — replaces band-power ratio)
            = mean Pearson r across the 4 spatial PhysNet channels.
            Real faces  : synchronized blood-flow   → high r  → high score
            Fake faces  : synthesis breaks spatial coherence → low r → low score
            Rationale   : FF++ deepfakes preserve temporal dynamics but often
                          disrupt the *spatial* phase coherence of the
                          cardiac signal across face regions.

        fft_score   (UNCHANGED from v1)
            = cardiac-band energy fraction from rolling rPPG accumulator.

        spectral_snr (NEW — logged only, not written to CSV)
            = tanh-normalised peak/noise-floor ratio in cardiac band.
    """

    _CLIP_LEN      = 30
    _STRIDE        = 15
    _FACE_SZ       = 128
    _FFT_ACCUM_LEN = 300
    _FFT_MIN_LEN   = 60

    def __init__(self, video_path: str, weight_path: str):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_path = Path(video_path).resolve()

        self.physnet = PhysNet(S=2).to(self.device)
        state_dict   = torch.load(weight_path, map_location=self.device)
        self.physnet.load_state_dict(state_dict)
        self.physnet.eval()
        logging.info(f"PhysNet loaded from {weight_path} on {self.device}")

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 → RGB float32 (128,128,3) in [0,1]."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self._FACE_SZ, self._FACE_SZ)).astype(np.float32) / 255.0

    def _to_tensor(self, frames: list) -> torch.Tensor:
        """List of (H,W,3) arrays → (1,3,T,H,W) tensor."""
        arr = np.stack(frames, axis=0).transpose(3, 0, 1, 2)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    @staticmethod
    def _znorm(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / (std if std > 1e-8 else 1.0)

    # ------------------------------------------------------------------
    # Score #1 – Spatial consistency  (PRIMARY discriminator)
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_consistency(block: np.ndarray) -> float:
        """
        block : (5, T) PhysNet output; channels 0-3 are the spatial signals.
        Returns mean Pearson r across the 6 channel pairs, clipped to [0,1].
        High  → real-face cardiac coherence.
        Low   → synthesis artefact or spatial phase break.
        """
        spatial = block[:4, :]          # (4, T)
        try:
            corr = np.corrcoef(spatial)                      # (4,4)
            vals = corr[np.triu_indices(4, k=1)]             # 6 off-diag values
            return float(np.clip(np.mean(vals), 0.0, 1.0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Score #2 – FFT cardiac-band energy fraction  (accumulator-based)
    # ------------------------------------------------------------------

    def _fft_score(self, accum: np.ndarray, fps: float) -> tuple:
        """Returns (score in [0,1], dominant HR bpm)."""
        if len(accum) < self._FFT_MIN_LEN:
            return 0.0, 0.0
        try:
            sig      = accum - np.linspace(accum[0], accum[-1], len(accum))
            filtered = butter_bandpass(sig, lowcut=0.6, highcut=4.0, fs=fps)
            windowed = filtered * np.hanning(len(filtered))
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs    = np.fft.rfftfreq(len(windowed), d=1.0 / fps)
            cmask    = (freqs >= 0.6) & (freqs <= 4.0)
            if not cmask.any():
                return 0.0, 0.0
            cs       = spectrum.copy(); cs[~cmask] = 0.0
            hr_bpm   = float(freqs[np.argmax(cs)] * 60.0)
            score    = float(np.clip(np.sum(spectrum[cmask]) /
                                     (np.sum(spectrum) + 1e-8), 0.0, 1.0))
            return score, hr_bpm
        except Exception:
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Score #3 – Spectral SNR  (logged only)
    # ------------------------------------------------------------------

    @staticmethod
    def _spectral_snr(signal: np.ndarray, fps: float) -> float:
        """peak cardiac-band power / noise-floor, tanh-normalised to [0,1]."""
        try:
            sig_n    = FusionEvaluator._znorm(signal)
            spectrum = np.abs(np.fft.rfft(sig_n * np.hanning(len(sig_n))))
            freqs    = np.fft.rfftfreq(len(sig_n), d=1.0 / fps)
            cmask    = (freqs >= 0.6) & (freqs <= 4.0)
            nmask    = (~cmask) & (freqs > 0)
            if not cmask.any() or not nmask.any():
                return 0.0
            peak  = float(np.max(spectrum[cmask]))
            noise = float(np.mean(spectrum[nmask]))
            return float(np.tanh(peak / (noise + 1e-8) / 3.0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> list:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_buf  = deque(maxlen=self._CLIP_LEN)
        rppg_accum = deque(maxlen=self._FFT_ACCUM_LEN)
        results    = []
        frame_cnt  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buf.append(self._preprocess(frame))
            frame_cnt += 1

            if len(frame_buf) == self._CLIP_LEN and frame_cnt % self._STRIDE == 0:
                clip = self._to_tensor(list(frame_buf))   # (1,3,30,128,128)
                with torch.no_grad():
                    rppg_block = self.physnet(clip)       # (1,5,30)

                block_np = rppg_block[0].cpu().numpy()   # (5,30)
                avg_sig  = block_np[-1]                  # (30,) spatial average

                # Score 1 – spatial consistency (PRIMARY)
                r_score = self._spatial_consistency(block_np)

                # Score 2 – FFT band energy from accumulator
                chunk = self._znorm(avg_sig[-self._STRIDE:])
                rppg_accum.extend(chunk.tolist())
                f_score, hr_bpm = self._fft_score(np.array(rppg_accum), fps)

                # Score 3 – spectral SNR (logged only)
                snr = self._spectral_snr(avg_sig, fps)

                results.append({
                    "window":     frame_cnt,
                    "rppg_score": round(r_score, 4),
                    "fft_score":  round(f_score,  4),
                })
                logging.info(
                    f"[window={frame_cnt:>4d}]  "
                    f"spatial_r={r_score:.4f}  "
                    f"fft={f_score:.4f}  "
                    f"snr={snr:.4f}  "
                    f"HR={hr_bpm:.1f}bpm  "
                    f"(accum={len(rppg_accum)})"
                )

        cap.release()
        return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   default="../datasets/FaceForensics/original_sequences/youtube/c23/videos/183.mp4")
    parser.add_argument("--output",  default="results/183_final_v2.csv")
    parser.add_argument("--weights", default="inference/model_weights.pt")
    args = parser.parse_args()

    ev   = FusionEvaluator(args.video, args.weights)
    data = ev.evaluate()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window", "rppg_score", "fft_score"])
        w.writeheader(); w.writerows(data)
    logging.info(f"Saved {args.output}  ({len(data)} windows)")
