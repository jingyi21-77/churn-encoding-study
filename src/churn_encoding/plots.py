"""Figures for the README / paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score

from .stability import top_q_mask

COLORS = {"onehot": "#2a6fdb", "target": "#e0782b"}
LS = {"logreg": "-", "gbt": "--"}
LABEL = {"onehot": "One-hot", "target": "Target", "logreg": "LogReg", "gbt": "GBT"}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _name(key):
    enc, mod = key
    return f"{LABEL[mod]} + {LABEL[enc]}"


def roc(scores: dict, y, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for key, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, LS[key[1]], color=COLORS[key[0]], lw=1.6,
                label=f"{_name(key)} (AUC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Out-of-fold ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def calibration(scores: dict, y, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for key, s in scores.items():
        frac, mean = calibration_curve(y, s, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, LS[key[1]], marker="o", ms=3, color=COLORS[key[0]], label=_name(key))
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate",
           title="Calibration (class-weighted models)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def score_scatter(scores: dict, path: Path, q: float = 0.10):
    models = sorted({k[1] for k in scores})
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        a, b = scores[("onehot", m)], scores[("target", m)]
        ma, mb = top_q_mask(a, q), top_q_mask(b, q)
        cat = np.where(ma & mb, "both", np.where(ma, "one-hot only", np.where(mb, "target only", "neither")))
        style = {"neither": ("#c8c8c8", 4), "both": ("#444444", 8),
                 "one-hot only": (COLORS["onehot"], 14), "target only": (COLORS["target"], 14)}
        for c in ["neither", "both", "one-hot only", "target only"]:
            sel = cat == c
            ax.scatter(a[sel], b[sel], s=style[c][1], c=style[c][0], alpha=0.7, lw=0,
                       label=f"{c} (n={sel.sum()})")
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
        ax.set(xlabel="P(churn) — one-hot", ylabel="P(churn) — target",
               title=f"{LABEL[m]}: who lands in the top {int(q*100)}%?")
        ax.legend(fontsize=7, loc="upper left", markerscale=1.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def stability_heatmap(table: pd.DataFrame, labels: list[str], metric: str, path: Path, title: str):
    n = len(labels)
    M = np.eye(n)
    idx = {l: i for i, l in enumerate(labels)}
    for _, r in table.iterrows():
        i, j = idx[r["a"]], idx[r["b"]]
        M[i, j] = M[j, i] = r[metric]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, vmin=min(0.5, M.min()), vmax=1, cmap="viridis")
    ax.set_xticks(range(n), labels, rotation=30, ha="right"); ax.set_yticks(range(n), labels)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.8 else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def power_curve(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(df["relative_mde"] * 100, df["n_total"], marker="o", color="#2a6fdb")
    for _, r in df.iterrows():
        ax.annotate(f"{int(r['n_total']):,}", (r["relative_mde"] * 100, r["n_total"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set(xlabel="Minimum detectable relative churn reduction (%)",
           ylabel="Total customers needed (both arms)",
           title=f"A/B sample size (baseline churn {df['p_control'].iloc[0]:.1%}, α=0.05, power=0.8)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
