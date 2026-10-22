"""Predictive metrics (paper §3.2): AUC, F1, precision, recall, Brier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (brier_score_loss, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises F1 on the given (validation) data."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def cost_benefit_threshold(offer_cost: float, value_saved: float, uplift: float) -> float:
    """Target a customer when  p * uplift * value_saved  >  offer_cost.

    ``uplift`` is the assumed probability that the offer prevents churn for a
    would-be churner. Returns the minimum churn probability worth targeting.
    """
    return offer_cost / (uplift * value_saved)


def fold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    yhat = (p >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y, p),
        "f1": f1_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
    }
