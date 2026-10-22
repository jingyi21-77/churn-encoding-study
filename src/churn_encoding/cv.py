"""Stratified 5-fold train / validation / test protocol (paper §3.2).

For outer iteration ``k`` the five stratified folds are used as

    test  = fold k
    val   = fold (k + 1) mod 5          (hyper-parameter tuning, early stopping,
                                          F1 threshold selection)
    train = remaining three folds        (model + encoder fitting)

so every customer appears in exactly one test fold and we obtain a single
out-of-sample risk score per customer for each encoder × model cell.
Encoders are always fitted on ``train`` only — validation and test rows
never contribute to the category statistics that encode them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from .metrics import best_f1_threshold, fold_metrics
from .pipelines import make_pipeline


@dataclass
class CellResult:
    encoder: str
    model: str
    seed: int
    oof_score: np.ndarray                    # out-of-fold P(churn) per customer
    fold_rows: list[dict] = field(default_factory=list)


def fold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [test for _, test in skf.split(np.zeros(len(y)), y)]
    for k in range(n_splits):
        test = folds[k]
        val = folds[(k + 1) % n_splits]
        train = np.concatenate([folds[j] for j in range(n_splits) if j not in (k, (k + 1) % n_splits)])
        yield k, train, val, test


def _grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in product(*(space[k] for k in keys))]


def _fit(pipe, X, y, model: str):
    if model == "gbt":
        # class weighting via sample weights (LR uses class_weight='balanced')
        pipe.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


def _tune_and_fit(encoder, model, space, Xtr, ytr, Xva, yva, seed, enc_kw):
    """Return the fitted pipeline with the best validation log-loss.

    For GBT, early stopping = choosing the number of boosting stages that
    minimises validation log-loss (via ``staged_predict_proba``), then
    truncating the ensemble to that many trees.
    """
    best = (np.inf, None, None)
    for params in _grid(space):
        pipe = _fit(make_pipeline(encoder, model, params, seed, **enc_kw), Xtr, ytr, model)
        if model == "gbt":
            Zva = pipe.named_steps["prep"].transform(Xva)
            gb = pipe.named_steps["model"]
            losses = [log_loss(yva, p[:, 1]) for p in gb.staged_predict_proba(Zva)]
            n_best = int(np.argmin(losses)) + 1
            loss = losses[n_best - 1]
            if loss < best[0]:
                gb.estimators_ = gb.estimators_[:n_best]
                gb.train_score_ = gb.train_score_[:n_best]
                gb.n_estimators = gb.n_estimators_ = n_best
                best = (loss, pipe, {**params, "n_estimators": n_best})
        else:
            loss = log_loss(yva, pipe.predict_proba(Xva)[:, 1])
            if loss < best[0]:
                best = (loss, pipe, params)
    return best[1], best[2]


def run_cell(X: pd.DataFrame, y: np.ndarray, encoder: str, model: str, space: dict,
             n_splits: int = 5, seed: int = 0, enc_kw: dict | None = None,
             verbose: bool = True) -> CellResult:
    enc_kw = enc_kw or {}
    res = CellResult(encoder, model, seed, oof_score=np.full(len(y), np.nan))
    for k, tr, va, te in fold_indices(y, n_splits, seed):
        pipe, params = _tune_and_fit(encoder, model, space,
                                     X.iloc[tr], y[tr], X.iloc[va], y[va], seed, enc_kw)
        p_val = pipe.predict_proba(X.iloc[va])[:, 1]
        p_te = pipe.predict_proba(X.iloc[te])[:, 1]
        thr = best_f1_threshold(y[va], p_val)
        res.oof_score[te] = p_te
        row = {"encoder": encoder, "model": model, "seed": seed, "fold": k,
               "threshold": thr, "n_features": len(pipe.named_steps["prep"].get_feature_names_out()),
               "params": params, **fold_metrics(y[te], p_te, thr)}
        res.fold_rows.append(row)
        if verbose:
            print(f"  [{encoder:6s}|{model:6s}|seed {seed}] fold {k}: "
                  f"AUC={row['auc']:.4f} F1={row['f1']:.4f} params={params}")
    assert not np.isnan(res.oof_score).any()
    return res
