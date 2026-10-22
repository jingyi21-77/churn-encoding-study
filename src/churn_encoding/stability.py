"""Targeting-stability analysis (paper §4, RQ2).

Given two out-of-sample risk-score vectors for the same customers, measure
how much the *targeting decision* changes:

* Spearman rank correlation of the full score vectors;
* Jaccard overlap of the top-q (default 10%) highest-risk sets;
* who is gained / lost, and how those customers differ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def top_q_mask(score: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Boolean mask of the ceil(q·n) highest scores (ties broken by index)."""
    n = len(score)
    k = int(np.ceil(q * n))
    order = np.lexsort((np.arange(n), -score))  # stable: high score first
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def compare(s1: np.ndarray, s2: np.ndarray, q: float = 0.10) -> dict:
    m1, m2 = top_q_mask(s1, q), top_q_mask(s2, q)
    return {
        "spearman": float(spearmanr(s1, s2).statistic),
        "jaccard_top": jaccard(m1, m2),
        "overlap_share": float(np.logical_and(m1, m2).sum() / m1.sum()),
        "n_top": int(m1.sum()),
        "n_only_first": int((m1 & ~m2).sum()),
        "n_only_second": int((m2 & ~m1).sum()),
    }


def pairwise_table(scores: dict[str, np.ndarray], q: float = 0.10) -> pd.DataFrame:
    """All pairwise comparisons between named score vectors."""
    names = list(scores)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"a": a, "b": b, **compare(scores[a], scores[b], q)})
    return pd.DataFrame(rows)


def segment_profile(X: pd.DataFrame, y: np.ndarray, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Descriptive stats of each targeted group (tenure, charges, contract mix…)."""
    rows = []
    for name, mask in groups.items():
        sub = X[mask]
        row = {
            "group": name, "n": int(mask.sum()),
            "actual_churn_rate": float(y[mask].mean()) if mask.any() else np.nan,
            "mean_tenure": sub["tenure"].mean(),
            "mean_monthly_charges": sub["MonthlyCharges"].mean(),
            "senior_share": (sub["SeniorCitizen"] == "Yes").mean(),
        }
        for lvl, share in sub["Contract"].value_counts(normalize=True).items():
            row[f"contract={lvl}"] = share
        for lvl, share in sub["PaymentMethod"].value_counts(normalize=True).items():
            row[f"payment={lvl}"] = share
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)
