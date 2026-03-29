#!/usr/bin/env python3
"""
analysis/compare_report.py

Side-by-side comparison of two batch-evaluation runs (v1 vs v2).
Generates four comparison figures and a printed diff-report.

Usage (from ~/im_lab/SoFake):
    # 1. produce v1 results (old fusion_evaluator on fix-frozen branch)
    #    already done → results/batch/

    # 2. pull improve-discrimination branch, re-run batch
    git fetch origin claude/improve-discrimination-p4LQK
    git checkout origin/claude/improve-discrimination-p4LQK \\
        -- evaluation/fusion_evaluator.py \\
           analysis/compare_report.py
    python3 evaluation/batch_evaluate.py --output results/batch_v2

    # 3. run comparison
    python3 analysis/compare_report.py \\
        --old results/batch   --old-label "v1 band-power" \\
        --new results/batch_v2 --new-label "v2 spatial-r" \\
        --output analysis/figures_compare
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ------------------------------------------------------------------ helpers

def load_df(results_root: Path) -> pd.DataFrame:
    df = pd.read_csv(results_root / "summary.csv")
    df = df[df["status"] == "ok"].copy()

    def _comp(p):
        for part in Path(p).parts:
            if part in ("c23", "c40", "raw"):
                return part
        return "unknown"

    df["compression"] = df["video"].apply(_comp)
    df["y_true"]      = (df["label"] == "fake").astype(int)
    # v1 used band-power (high rppg = real), v2 uses spatial-r (high rppg = real)
    # In both cases: fake_score = 1 - mean_rppg
    df["fake_score"] = 1.0 - df["mean_rppg"]
    return df


def _auc_manual(y, s):
    thrs = np.sort(np.unique(s))[::-1]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0: return float("nan")
    tprs, fprs = [0.], [0.]
    for t in thrs:
        p  = (s >= t).astype(int)
        tprs.append(((p==1)&(y==1)).sum() / pos)
        fprs.append(((p==1)&(y==0)).sum() / neg)
    tprs.append(1.); fprs.append(1.)
    return float(np.trapz(tprs, fprs))


def auc(y, s):
    if HAS_SKLEARN and len(np.unique(y)) == 2:
        return roc_auc_score(y, s)
    return _auc_manual(np.asarray(y), np.asarray(s))


def roc(y, s):
    if HAS_SKLEARN:
        return roc_curve(y, s)
    thrs = np.sort(np.unique(s))[::-1]
    pos, neg = y.sum(), len(y) - y.sum()
    tprs, fprs = [0.], [0.]
    for t in thrs:
        p = (s >= t).astype(int)
        tprs.append(((p==1)&(y==1)).sum() / pos)
        fprs.append(((p==1)&(y==0)).sum() / neg)
    tprs.append(1.); fprs.append(1.)
    return np.array(fprs), np.array(tprs), thrs


def best_f1(y, s):
    if HAS_SKLEARN:
        pr, rc, thr = precision_recall_curve(y, s)
        f1s = np.where((pr+rc)>0, 2*pr*rc/(pr+rc), 0)
        i   = np.argmax(f1s)
        return float(f1s[i]), float(thr[i]) if i < len(thr) else 0.5
    best, bt = 0., 0.5
    for t in np.linspace(0, 1, 101):
        pred = (np.asarray(s) >= t).astype(int)
        tp = ((pred==1)&(np.asarray(y)==1)).sum()
        fp = ((pred==1)&(np.asarray(y)==0)).sum()
        fn = ((pred==0)&(np.asarray(y)==1)).sum()
        f = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
        if f > best: best, bt = f, t
    return best, bt


def far95(y, s):
    fpr_a, tpr_a, _ = roc(y, s)
    mask = np.asarray(tpr_a) >= 0.95
    return float(np.asarray(fpr_a)[mask][0]) if mask.any() else float("nan")


def metrics_for(df):
    """Return dict with overall + per-compression metrics."""
    out = {}
    for name, sub in [("overall", df),
                       ("c23",    df[df.compression=="c23"]),
                       ("c40",    df[df.compression=="c40"])]:
        y, s = sub["y_true"].values, sub["fake_score"].values
        if len(sub) < 2 or len(np.unique(y)) < 2:
            out[name] = dict(n=len(sub), auc=float("nan"),
                             f1=float("nan"), far=float("nan"))
            continue
        a           = auc(y, s)
        f, _        = best_f1(y, s)
        fa          = far95(y, s)
        out[name]   = dict(n=len(sub), auc=a, f1=f, far=fa)
    return out


def signal_loss(results_root: Path, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        p = results_root / Path(row["video"]).with_suffix(".csv")
        if not p.exists(): continue
        wdf  = pd.read_csv(p)
        tot  = len(wdf)
        lost = (wdf["fft_score"] == 0).sum()
        rows.append({"video": row["video"], "label": row["label"],
                     "compression": row["compression"],
                     "loss_rate": lost/tot if tot else 0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots

def plot_roc_compare(old_df, new_df, old_lbl, new_lbl, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    subsets = [("Overall", old_df, new_df),
               ("c23",  old_df[old_df.compression=="c23"], new_df[new_df.compression=="c23"]),
               ("c40",  old_df[old_df.compression=="c40"], new_df[new_df.compression=="c40"])]

    for ax, (title, osub, nsub) in zip(axes, subsets):
        ax.plot([0,1],[0,1],"k--",lw=0.8,label="Random")
        for sub, lbl, color in [(osub, old_lbl, "#e74c3c"), (nsub, new_lbl, "#2ecc71")]:
            if len(sub)<2 or sub["y_true"].nunique()<2: continue
            fp, tp, _ = roc(sub["y_true"].values, sub["fake_score"].values)
            a = auc(sub["y_true"].values, sub["fake_score"].values)
            ax.plot(fp, tp, color=color, lw=2, label=f"{lbl}  AUC={a:.3f}")
        ax.set_title(f"ROC – {title}"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir/"fig_compare_roc.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_compare_roc.png")


def plot_metrics_bar(old_m, new_m, old_lbl, new_lbl, fig_dir):
    subsets  = ["overall", "c23", "c40"]
    metrics  = ["auc", "f1", "far"]
    m_labels = ["AUC", "F1", "FAR@TPR95"]
    colors   = {"auc": "#3498db", "f1": "#2ecc71", "far": "#e74c3c"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ss in zip(axes, subsets):
        x = np.arange(len(metrics))
        w = 0.35
        for i, (m, ml) in enumerate(zip(metrics, m_labels)):
            ov = old_m[ss].get(m, float("nan"))
            nv = new_m[ss].get(m, float("nan"))
            b1 = ax.bar(i - w/2, ov if not np.isnan(ov) else 0,
                        w, color=colors[m], alpha=0.5, label=f"{old_lbl}" if i==0 else "")
            b2 = ax.bar(i + w/2, nv if not np.isnan(nv) else 0,
                        w, color=colors[m], alpha=0.95, label=f"{new_lbl}" if i==0 else "")
            for b, v in [(b1, ov), (b2, nv)]:
                if not np.isnan(v):
                    ax.text(b[0].get_x()+b[0].get_width()/2,
                            b[0].get_height()+0.01, f"{v:.3f}",
                            ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(m_labels)
        ax.set_ylim(0, 1.2); ax.set_title(f"Metrics – {ss.upper()}")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        # Annotate AUC delta
        d = new_m[ss].get("auc", float("nan")) - old_m[ss].get("auc", float("nan"))
        if not np.isnan(d):
            sign = "+" if d >= 0 else ""
            col  = "green" if d >= 0 else "red"
            ax.text(0.98, 0.97, f"ΔAUC={sign}{d:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    color=col, fontweight="bold", fontsize=10,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    fig.suptitle(f"Metrics comparison: {old_lbl} vs {new_lbl}", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir/"fig_compare_metrics_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_compare_metrics_bar.png")


def plot_score_dist_compare(old_df, new_df, old_lbl, new_lbl, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    titles = [(old_lbl + " c23", old_df[old_df.compression=="c23"]),
              (new_lbl + " c23", new_df[new_df.compression=="c23"]),
              (old_lbl + " c40", old_df[old_df.compression=="c40"]),
              (new_lbl + " c40", new_df[new_df.compression=="c40"])]
    colors = {"real": "#2ecc71", "fake": "#e74c3c"}
    for ax, (title, sub) in zip(axes.flat, titles):
        for lbl in ["real", "fake"]:
            v = sub[sub.label==lbl]["mean_rppg"].values
            if len(v): ax.hist(v, bins=10, alpha=0.65, color=colors[lbl],
                               label=lbl, density=True)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("mean_rppg_score"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Score distribution  (better: real & fake peaks clearly separated)")
    fig.tight_layout()
    fig.savefig(fig_dir/"fig_compare_score_dist.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_compare_score_dist.png")


def plot_stability_compare(old_stab, new_stab, old_lbl, new_lbl, fig_dir):
    if old_stab.empty or new_stab.empty: return
    merged = old_stab[["video","loss_rate"]].rename(columns={"loss_rate":"old"}).merge(
             new_stab[["video","loss_rate"]].rename(columns={"loss_rate":"new"}),
             on="video")
    merged["label"] = old_stab.set_index("video").loc[merged.video, "label"].values

    fig, ax = plt.subplots(figsize=(max(8, len(merged)*0.7), 5))
    x = np.arange(len(merged))
    w = 0.35
    colors = merged["label"].map({"real":"#2ecc71","fake":"#e74c3c"}).values
    ax.bar(x - w/2, merged["old"].values, w, color=colors, alpha=0.45, label=old_lbl)
    ax.bar(x + w/2, merged["new"].values, w, color=colors, alpha=0.90, label=new_lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([Path(v).stem for v in merged["video"]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Signal loss rate  (fft_score==0)")
    ax.set_title("rPPG Signal Stability: old vs new\n(green=real, red=fake)")
    handles = [mpatches.Patch(fc="grey", alpha=0.45, label=old_lbl),
               mpatches.Patch(fc="grey", alpha=0.90, label=new_lbl)]
    ax.legend(handles=handles); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir/"fig_compare_stability.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_compare_stability.png")


# ------------------------------------------------------------------ report

def print_diff(old_m, new_m, old_lbl, new_lbl):
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  COMPARISON REPORT: {old_lbl}  vs  {new_lbl}")
    print(sep)
    print(f"  {'Subset':<10} {'Metric':<12} {old_lbl:>18} {new_lbl:>18} {'Delta':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*18} {'-'*18} {'-'*10}")
    for ss in ["overall", "c23", "c40"]:
        for m, better in [("auc","up"),("f1","up"),("far","down")]:
            ov = old_m[ss].get(m, float("nan"))
            nv = new_m[ss].get(m, float("nan"))
            d  = nv - ov if not (np.isnan(ov) or np.isnan(nv)) else float("nan")
            sign = "+" if d > 0 else ""
            improved = (d > 0 and better=="up") or (d < 0 and better=="down")
            flag = "\u2713" if improved else ("\u2717" if not np.isnan(d) else "")
            ov_s = f"{ov:.4f}" if not np.isnan(ov) else "n/a"
            nv_s = f"{nv:.4f}" if not np.isnan(nv) else "n/a"
            d_s  = f"{sign}{d:.4f} {flag}" if not np.isnan(d) else "n/a"
            print(f"  {ss:<10} {m:<12} {ov_s:>18} {nv_s:>18} {d_s:>10}")
    print(sep)

    # RAM fix note
    try:
        from model.physnet_model import PhysNet
        import torch
        n_params = sum(p.numel() for p in PhysNet(S=2).parameters())
        ram_fp32 = n_params * 4 / 1e6
        print(f"\n  [RAM fix] PhysNet footprint: {n_params:,} params = "
              f"{ram_fp32:.1f} MB fp32  (tracemalloc in v1 was buggy — "
              f"measured 0 because model loads before timer starts)")
    except Exception:
        pass
    print()


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old",       default="results/batch")
    ap.add_argument("--old-label", default="v1 band-power")
    ap.add_argument("--new",       default="results/batch_v2")
    ap.add_argument("--new-label", default="v2 spatial-r")
    ap.add_argument("--output",    default="analysis/figures_compare")
    args = ap.parse_args()

    old_root = Path(args.old)
    new_root = Path(args.new)
    fig_dir  = Path(args.output)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading v1  : {old_root}")
    old_df = load_df(old_root)
    print(f"Loading v2  : {new_root}")
    new_df = load_df(new_root)

    old_m = metrics_for(old_df)
    new_m = metrics_for(new_df)

    old_stab = signal_loss(old_root, old_df)
    new_stab = signal_loss(new_root, new_df)

    print("\nGenerating comparison figures:")
    plot_roc_compare(old_df, new_df, args.old_label, args.new_label, fig_dir)
    plot_metrics_bar(old_m,  new_m,  args.old_label, args.new_label, fig_dir)
    plot_score_dist_compare(old_df, new_df, args.old_label, args.new_label, fig_dir)
    plot_stability_compare(old_stab, new_stab, args.old_label, args.new_label, fig_dir)

    print_diff(old_m, new_m, args.old_label, args.new_label)


if __name__ == "__main__":
    main()
