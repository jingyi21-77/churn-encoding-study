"""Encoder × model pipelines (paper §3.1–3.2).

Every pipeline shares the same numeric handling and the same model
hyper-parameter space; only the categorical encoder changes. That is the
controlled factor of the study.
"""
from __future__ import annotations

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from .data import CATEGORICAL_COLS, NUMERIC_COLS

_SKLEARN_GE_1_9 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 9)

ENCODERS = ("onehot", "target")
MODELS = ("logreg", "gbt")


def make_encoder(name: str, smooth="auto", inner_cv: int = 5, random_state: int = 0):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "target":
        # sklearn's TargetEncoder uses *internal cross-fitting* in
        # fit_transform: each training row is encoded with statistics from
        # the other inner folds, so a row never sees its own label. At
        # transform time (validation/test folds) it uses statistics fitted
        # on the full training portion only. Smoothing shrinks rare levels
        # toward the global churn rate (empirical-Bayes style).
        if _SKLEARN_GE_1_9:
            kw = {"cv": StratifiedKFold(inner_cv, shuffle=True, random_state=random_state)}
        else:
            kw = {"cv": inner_cv, "shuffle": True, "random_state": random_state}
        return TargetEncoder(target_type="binary", smooth=smooth, **kw)
    raise ValueError(f"unknown encoder {name!r}")


def make_preprocessor(encoder: str, model: str, **enc_kw) -> ColumnTransformer:
    cat = make_encoder(encoder, **enc_kw)
    if model == "logreg":
        # Scale numerics (and target-encoded columns) so L2 penalises
        # coefficients on a comparable scale.
        cat = Pipeline([("enc", cat), ("scale", StandardScaler())]) if encoder == "target" else cat
        num = StandardScaler()
    else:
        num = "passthrough"  # trees are scale-invariant
    return ColumnTransformer(
        [("num", num, NUMERIC_COLS), ("cat", cat, CATEGORICAL_COLS)],
        verbose_feature_names_out=False,
    )


def make_model(model: str, params: dict, random_state: int = 0):
    if model == "logreg":
        return LogisticRegression(
            C=params.get("C", 1.0), l1_ratio=0.0, solver="lbfgs",
            class_weight="balanced", max_iter=5000,
        )
    if model == "gbt":
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            random_state=random_state,
        )
    raise ValueError(f"unknown model {model!r}")


def make_pipeline(encoder: str, model: str, params: dict, random_state: int = 0, **enc_kw) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(encoder, model, random_state=random_state, **enc_kw)),
        ("model", make_model(model, params, random_state=random_state)),
    ])
