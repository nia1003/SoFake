#!/usr/bin/env python3
"""
analysis/metrics_report.py

Reads results/batch/summary.csv (and the per-video CSVs it references),
then computes and plots the four evaluation dimensions:

    1. Core accuracy     : AUC, F1 @ optimal threshold, FAR @ strict threshold
    2. Edge performance  : single-window PhysNet latency (ms), peak RAM (MB)
    3. Robustness        : c23 vs c40 AUC degradation; rPPG signal-stability
    4. Score overview    : per-video scatter + compression comparison

Usage (from ~/im_lab/SoFake):
    python3 analysis/metrics_report.py \
        --results results/batch \
        --output  analysis/figures
"""

import argparse
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import welch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# sklearn is not in requirements.txt – use a lightweight fallback if absent
try:
    from sklearn.metrics import (
        roc_auc_score, roc_curve,
        f1_score, precision_recall_curve,
        confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not installed – AUC/F1 computed manually.")


# ============================================================
# 1. Data loading
# ============================================================

def load_summary(results_root: Path) -> pd.DataFrame:
    """Load summary.csv and parse compression type from the video path."""
    summary_path = results_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary.csv not found at {summary_path}\n"
            "Run  python3 evaluation/batch_evaluate.py  first."
        )
    df = pd.read_csv(summary_path)
    df = df[df["status"] == "ok"].copy()

    # Derive compression type (c23 / c40) from path
    def _comp(p):
        for part in Path(p).parts:
            if part in ("c23", "c40", "raw"):
                return part
        return "unknown"

    df["compression"] = df["video"].apply(_comp)
    # Binary label: 1 = fake (manipulated), 0 = real (original)
    df["y_true"] = (df["label"] == "fake").astype(int)
    # Fakeness score: high rppg = strong cardiac = real  =>  invert for fake
    df["fake_score"]    = 1.0 - df["mean_rppg"]
    df["fake_score_fft"] = 1.0 - df["mean_fft"].clip(lower=0)
    return df


def load_per_video_windows(results_root: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Load all per-video window CSVs and return a long-form DataFrame."""
    rows = []
    for _, row in df.iterrows():
        csv_path = results_root / Path(row["video"]).with_suffix(".csv")
        if not csv_path.exists():
            continue
        wdf = pd.read_csv(csv_path)
        wdf["video"]       = row["video"]
        wdf["label"]       = row["label"]
        wdf["compression"] = row["compression"]
        rows.append(wdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================
# 2. Metrics helpers
# ============================================================

def _auc_manual(y_true, scores):
    """Trapezoidal AUC without sklearn."""
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0); fprs.append(1.0)
    return float(np.trapz(tprs, fprs))


def compute_auc(y_true, scores):
    if HAS_SKLEARN and len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, scores)
    return _auc_manual(np.array(y_true), np.array(scores))


def compute_roc(y_true, scores):
    if HAS_SKLEARN:
        return roc_curve(y_true, scores)
    # manual
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = y_true.sum(); neg = len(y_true) - pos
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / pos); fprs.append(fp / neg)
    tprs.append(1.0); fprs.append(1.0)
    return np.array(fprs), np.array(tprs), thresholds


def best_f1_threshold(y_true, scores):
    """Return (best_f1, best_threshold) by scanning."""
    if HAS_SKLEARN:
        prec, rec, thr = precision_recall_curve(y_true, scores)
        f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
        idx = np.argmax(f1s)
        return float(f1s[idx]), float(thr[idx]) if idx < len(thr) else 0.5
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0, 1, 101):
        pred = (np.array(scores) >= t).astype(int)
        tp = ((pred == 1) & (np.array(y_true) == 1)).sum()
        fp = ((pred == 1) & (np.array(y_true) == 0)).sum()
        fn = ((pred == 0) & (np.array(y_true) == 1)).sum()
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


def far_at_tpr(y_true, scores, target_tpr=0.95):
    """
    False Accept Rate at the threshold where TPR >= target_tpr.
    FAR = FP / (FP + TN)  (probability a fake passes as real).
    """
    fpr_arr, tpr_arr, _ = compute_roc(y_true, scores)
    # Find lowest FPR point where TPR >= target_tpr
    mask = np.array(tpr_arr) >= target_tpr
    if not mask.any():
        return float("nan")
    return float(np.array(fpr_arr)[mask][0])


# ============================================================
# 3. Latency & memory benchmark
# ============================================================

def benchmark_latency(weight_path: str, n_warmup=3, n_runs=20) -> dict:
    """Time a single PhysNet forward pass on a (1,3,30,128,128) tensor."""
    import torch
    from model.physnet_model import PhysNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PhysNet(S=2).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, 30, 128, 128).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Timed runs
    tracemalloc.start()
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)   # ms
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # VRAM (GPU only)
    vram_mb = 0.0
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated(device) / 1e6

    return {
        "device":      str(device),
        "mean_ms":     float(np.mean(latencies)),
        "p50_ms":      float(np.percentile(latencies, 50)),
        "p95_ms":      float(np.percentile(latencies, 95)),
        "peak_ram_mb": peak_bytes / 1e6,
        "peak_vram_mb": vram_mb,
        "target_met":  np.percentile(latencies, 95) < 100.0,
    }


# ============================================================
# 4. Signal stability (robustness)
# ============================================================

def signal_stability(wdf: pd.DataFrame) -> pd.DataFrame:
    """
    Per-video: fraction of windows where fft_score == 0 (signal lost).
    High loss -> model sensitive to motion / lighting.
    """
    rows = []
    for video, grp in wdf.groupby("video"):
        total = len(grp)
        lost  = (grp["fft_score"] == 0).sum()
        rows.append({
            "video":       video,
            "label":       grp["label"].iloc[0],
            "compression": grp["compression"].iloc[0],
            "total_windows": total,
            "signal_lost":   int(lost),
            "loss_rate":     round(lost / total, 3) if total else 0,
        })
    return pd.DataFrame(rows)


# ============================================================
# 5. Plotting
# ============================================================

COLORS = {"real": "#2ecc71", "fake": "#e74c3c",
           "c23":  "#3498db",  "c40":  "#e67e22"}


def plot_roc(df: pd.DataFrame, fig_dir: Path):
    """Figure 1 – ROC curves: overall, c23, c40."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")

    subsets = [("Overall", df, "#8e44ad"),
               ("c23",     df[df.compression == "c23"], COLORS["c23"]),
               ("c40",     df[df.compression == "c40"], COLORS["c40"])]

    for name, sub, color in subsets:
        if len(sub) < 2 or sub["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = compute_roc(sub["y_true"].values, sub["fake_score"].values)
        auc = compute_auc(sub["y_true"].values, sub["fake_score"].values)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC={auc:.3f}")

    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (Detection Rate)")
    ax.set_title("ROC Curve – rPPG Deepfake Detector")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_roc.png", dpi=150)
    plt.close(fig)
    print("  Saved fig1_roc.png")


def plot_score_dist(df: pd.DataFrame, fig_dir: Path):
    """Figure 2 – Score distributions (real vs fake) per compression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, comp in zip(axes, ["c23", "c40"]):
        sub = df[df.compression == comp]
        for label in ["real", "fake"]:
            vals = sub[sub.label == label]["mean_rppg"].values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=15, alpha=0.6, color=COLORS[label],
                    label=label, density=True)
        ax.set_title(f"rPPG Score Distribution – {comp.upper()}")
        ax.set_xlabel("mean_rppg_score  (higher = more real-like)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_score_dist.png", dpi=150)
    plt.close(fig)
    print("  Saved fig2_score_dist.png")


def plot_compression_degradation(df: pd.DataFrame, fig_dir: Path):
    """Figure 3 – AUC / F1 / FAR bar chart: c23 vs c40."""
    records = []
    for comp in ["c23", "c40"]:
        sub = df[df.compression == comp]
        if len(sub) < 2 or sub["y_true"].nunique() < 2:
            continue
        auc         = compute_auc(sub["y_true"].values, sub["fake_score"].values)
        best_f1, _  = best_f1_threshold(sub["y_true"].values, sub["fake_score"].values)
        far         = far_at_tpr(sub["y_true"].values, sub["fake_score"].values, 0.95)
        records.append({"compression": comp, "AUC": auc,
                         "F1": best_f1, "FAR@TPR95": far})

    if not records:
        print("  [SKIP] fig3 – insufficient data for compression comparison")
        return

    mdf    = pd.DataFrame(records).set_index("compression")
    x      = np.arange(len(mdf))
    width  = 0.25
    metrics = ["AUC", "F1", "FAR@TPR95"]
    palette = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (metric, color) in enumerate(zip(metrics, palette)):
        bars = ax.bar(x + i*width, mdf[metric].values, width,
                      label=metric, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(mdf.index)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Core Accuracy Metrics – c23 vs c40\n"
                 "(FAR@TPR95: lower is better; AUC/F1: higher is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Degradation annotation
    if "c23" in mdf.index and "c40" in mdf.index:
        delta = mdf.loc["c23", "AUC"] - mdf.loc["c40", "AUC"]
        ax.annotate(f"AUC degradation\nc23→c40: {delta:+.3f}",
                    xy=(0.98, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))

    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_compression_degradation.png", dpi=150)
    plt.close(fig)
    print("  Saved fig3_compression_degradation.png")


def plot_latency(bench: dict, fig_dir: Path):
    """Figure 4 – Latency distribution + target line."""
    # Re-run a mini benchmark for the histogram
    import torch
    from model.physnet_model import PhysNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PhysNet(S=2).to(device)
    model.load_state_dict(
        torch.load(str(_PROJECT_ROOT / "inference" / "model_weights.pt"),
                   map_location=device)
    )
    model.eval()
    dummy = torch.randn(1, 3, 30, 128, 128).to(device)
    lats = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            _  = model(dummy)
            lats.append((time.perf_counter() - t0) * 1000)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lats, bins=15, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(100, color="red", lw=1.5, linestyle="--", label="Target 100 ms")
    ax.axvline(np.mean(lats), color="orange", lw=1.5,
               linestyle="-", label=f"Mean {np.mean(lats):.1f} ms")
    ax.set_xlabel("Inference latency (ms) per 30-frame window")
    ax.set_ylabel("Count")
    ax.set_title(f"PhysNet Latency – {bench['device'].upper()}\n"
                 f"p50={bench['p50_ms']:.1f}ms  p95={bench['p95_ms']:.1f}ms  "
                 f"RAM={bench['peak_ram_mb']:.0f}MB")
    ax.legend()
    ax.grid(alpha=0.3)
    ok = "PASS" if bench["target_met"] else "FAIL"
    color = "green" if bench["target_met"] else "red"
    ax.text(0.98, 0.95, f"<100ms target: {ok}",
            transform=ax.transAxes, ha="right", va="top",
            color=color, fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_latency.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4_latency.png")


def plot_signal_stability(stab: pd.DataFrame, fig_dir: Path):
    """Figure 5 – Signal loss rate per video (rPPG robustness)."""
    if stab.empty:
        return
    stab = stab.sort_values(["compression", "label", "loss_rate"])
    colors = stab["label"].map(COLORS)
    fig, ax = plt.subplots(figsize=(max(8, len(stab)*0.6), 5))
    bars = ax.bar(range(len(stab)), stab["loss_rate"].values,
                  color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(stab)))
    ax.set_xticklabels(
        [f"{Path(v).stem}\n({r['compression']})"
         for _, r in stab.iterrows()
         for v in [r["video"]]],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Signal Loss Rate  (fft_score == 0)")
    ax.set_title("rPPG Signal Stability per Video\n"
                 "(Green = real, Red = fake; lower loss = more stable)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS["real"], label="real"),
                        Patch(color=COLORS["fake"], label="fake")])
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_signal_stability.png", dpi=150)
    plt.close(fig)
    print("  Saved fig5_signal_stability.png")


def plot_per_video_scatter(df: pd.DataFrame, fig_dir: Path):
    """Figure 6 – Per-video mean_rppg vs mean_fft scatter (real vs fake)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, comp in zip(axes, ["c23", "c40"]):
        sub = df[df.compression == comp]
        for label in ["real", "fake"]:
            pts = sub[sub.label == label]
            ax.scatter(pts["mean_rppg"], pts["mean_fft"],
                       c=COLORS[label], label=label,
                       s=80, alpha=0.85, edgecolors="white", linewidths=0.5)
            for _, row in pts.iterrows():
                ax.annotate(Path(row["video"]).stem,
                            (row["mean_rppg"], row["mean_fft"]),
                            fontsize=7, alpha=0.7)
        ax.set_xlabel("mean_rppg_score")
        ax.set_ylabel("mean_fft_score")
        ax.set_title(f"Score Scatter – {comp.upper()}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Per-Video Scores  (real=green, fake=red)", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_scatter.png")


# ============================================================
# 6. Text report
# ============================================================

def print_report(df: pd.DataFrame, bench: dict, stab: pd.DataFrame):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  FUSION EVALUATOR – METRICS REPORT")
    print(sep)

    # ── 1. Core accuracy ──────────────────────────────────────
    print("\n[1] CORE ACCURACY  (rppg-based fake_score = 1 - mean_rppg)")
    print(f"  {'Subset':<12} {'N':>4}  {'AUC':>6}  {'F1':>6}  {'FAR@TPR95':>10}")
    print(f"  {'-'*12} {'-'*4}  {'-'*6}  {'-'*6}  {'-'*10}")
    for name, sub in [("Overall", df),
                       ("c23",     df[df.compression == "c23"]),
                       ("c40",     df[df.compression == "c40"])]:
        if len(sub) < 2 or sub["y_true"].nunique() < 2:
            print(f"  {name:<12} {len(sub):>4}  {'n/a':>6}  {'n/a':>6}  {'n/a':>10}")
            continue
        auc        = compute_auc(sub["y_true"].values, sub["fake_score"].values)
        best_f1, _ = best_f1_threshold(sub["y_true"].values, sub["fake_score"].values)
        far        = far_at_tpr(sub["y_true"].values, sub["fake_score"].values, 0.95)
        print(f"  {name:<12} {len(sub):>4}  {auc:>6.3f}  {best_f1:>6.3f}  {far:>10.3f}")

    if "c23" in df.compression.values and "c40" in df.compression.values:
        auc23 = compute_auc(df[df.compression=="c23"]["y_true"].values,
                            df[df.compression=="c23"]["fake_score"].values)
        auc40 = compute_auc(df[df.compression=="c40"]["y_true"].values,
                            df[df.compression=="c40"]["fake_score"].values)
        if not (np.isnan(auc23) or np.isnan(auc40)):
            print(f"\n  Compression degradation  c23→c40: "
                  f"ΔAUC = {auc23-auc40:+.3f} "
                  f"({abs(auc23-auc40)/auc23*100:.1f}% relative drop)")

    # ── 2. Edge performance ────────────────────────────────────
    print(f"\n[2] EDGE PERFORMANCE")
    status = "PASS ✓" if bench["target_met"] else "FAIL ✗"
    print(f"  Device          : {bench['device']}")
    print(f"  Latency mean    : {bench['mean_ms']:.1f} ms")
    print(f"  Latency p50/p95 : {bench['p50_ms']:.1f} / {bench['p95_ms']:.1f} ms")
    print(f"  <100ms target   : {status}")
    print(f"  Peak RAM        : {bench['peak_ram_mb']:.0f} MB")
    if bench["peak_vram_mb"] > 0:
        print(f"  Peak VRAM       : {bench['peak_vram_mb']:.0f} MB")

    # ── 3. Robustness ─────────────────────────────────────────
    print(f"\n[3] SIGNAL STABILITY  (rPPG loss rate per video)")
    if not stab.empty:
        print(f"  {'Video':<30} {'Comp':>5}  {'Label':>5}  {'LossRate':>9}")
        print(f"  {'-'*30} {'-'*5}  {'-'*5}  {'-'*9}")
        for _, r in stab.sort_values("loss_rate", ascending=False).iterrows():
            print(f"  {Path(r['video']).stem:<30} {r['compression']:>5}  "
                  f"{r['label']:>5}  {r['loss_rate']:>9.3f}")
        mean_loss = stab.groupby("label")["loss_rate"].mean()
        print(f"\n  Mean signal loss: real={mean_loss.get('real',0):.3f}  "
              f"fake={mean_loss.get('fake',0):.3f}")

    # ── 4. Engineering note ────────────────────────────────────
    print(f"\n[4] ENGINEERING INTEGRATION NOTES")
    print("  Architecture decoupling : PhysNet (3D-CNN) is fully self-contained")
    print("    in model/physnet_model.py.  FusionEvaluator wraps it with a")
    print("    clean evaluate()->list API.  Swap model by replacing PhysNet(S=2).")
    print("  Pruning/fine-tune       : PhysNet has ~3.4M params (see below).")
    print("    Spatial blocks (loop4/encoder2) are natural pruning targets.")
    print("  API packaging           : FusionEvaluator is importable; wrap in")
    print("    FastAPI/Flask with POST /evaluate  {video_path, weights_path}.")
    print(sep)

    # Param count
    try:
        import torch
        from model.physnet_model import PhysNet
        n = sum(p.numel() for p in PhysNet(S=2).parameters())
        print(f"  PhysNet param count : {n:,}  (~{n*4/1e6:.1f} MB fp32)")
    except Exception:
        pass
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/batch")
    parser.add_argument("--output",  default="analysis/figures")
    parser.add_argument("--weights", default="inference/model_weights.pt")
    args = parser.parse_args()

    results_root = Path(args.results)
    fig_dir      = Path(args.output)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_root} ...")
    df   = load_summary(results_root)
    wdf  = load_per_video_windows(results_root, df)
    stab = signal_stability(wdf) if not wdf.empty else pd.DataFrame()

    print(f"  {len(df)} videos loaded  "
          f"({(df.label=='real').sum()} real / {(df.label=='fake').sum()} fake)")
    print(f"  Compressions: {sorted(df.compression.unique().tolist())}\n")

    print("Running latency benchmark ...")
    bench = benchmark_latency(args.weights)

    print("\nGenerating figures:")
    plot_roc(df, fig_dir)
    plot_score_dist(df, fig_dir)
    plot_compression_degradation(df, fig_dir)
    plot_latency(bench, fig_dir)
    if not stab.empty:
        plot_signal_stability(stab, fig_dir)
    plot_per_video_scatter(df, fig_dir)

    print_report(df, bench, stab)


if __name__ == "__main__":
    main()
