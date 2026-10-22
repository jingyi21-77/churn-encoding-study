import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn_encoding import abtest, stability  # noqa: E402
from churn_encoding.cv import fold_indices  # noqa: E402
from churn_encoding.data import CATEGORICAL_COLS, NUMERIC_COLS, clean  # noqa: E402
from churn_encoding.pipelines import make_pipeline  # noqa: E402


# ---------------------------------------------------------------- fixtures
def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["A", "B", "C"], n) for c in CATEGORICAL_COLS})
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(20, 110, n).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    logit = -1 + 1.5 * (df["Contract"] == "A") - 0.03 * df["tenure"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return clean(df)[NUMERIC_COLS + CATEGORICAL_COLS], y


# ---------------------------------------------------------------- data
def test_clean_blank_total_charges_become_zero():
    X, _ = _toy()
    assert X["TotalCharges"].dtype.kind == "f"
    assert (X.loc[X["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(X["SeniorCitizen"]) <= {"Yes", "No"}


# ---------------------------------------------------------------- CV protocol
def test_fold_indices_partition_and_disjoint():
    y = np.r_[np.zeros(400), np.ones(100)].astype(int)
    seen_test = []
    for _, tr, va, te in fold_indices(y, 5, seed=0):
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == len(y)
        assert abs(y[te].mean() - 0.2) < 0.02  # stratified
        seen_test.extend(te)
    assert sorted(seen_test) == list(range(len(y)))  # every row tested exactly once


# ---------------------------------------------------------------- leakage
def test_target_encoder_does_not_leak_labels():
    """With random labels, a leakage-free target-encoded model must score ~0.5 AUC
    on held-out data, even with a high-cardinality ID-like column."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    X, _ = _toy(1500, seed=1)
    X["PaymentMethod"] = [f"id{i}" for i in rng.integers(0, 700, len(X))]  # near-unique levels
    y = rng.integers(0, 2, len(X))
    pipe = make_pipeline("target", "gbt", {"n_estimators": 100, "max_depth": 3})
    pipe.fit(X.iloc[:1000], y[:1000])
    auc = roc_auc_score(y[1000:], pipe.predict_proba(X.iloc[1000:])[:, 1])
    assert 0.40 < auc < 0.60


@pytest.mark.parametrize("enc", ["onehot", "target"])
@pytest.mark.parametrize("model", ["logreg", "gbt"])
def test_pipelines_fit_and_beat_chance(enc, model):
    from sklearn.metrics import roc_auc_score
    X, y = _toy(1200)
    pipe = make_pipeline(enc, model, {"C": 1.0, "n_estimators": 60, "max_depth": 2})
    pipe.fit(X.iloc[:800], y[:800])
    assert roc_auc_score(y[800:], pipe.predict_proba(X.iloc[800:])[:, 1]) > 0.6


# ---------------------------------------------------------------- stability
def test_top_q_and_jaccard():
    s = np.arange(100, dtype=float)
    m = stability.top_q_mask(s, 0.10)
    assert m.sum() == 10 and m[-10:].all()
    assert stability.jaccard(m, m) == 1.0
    r = stability.compare(s, s[::-1])
    assert r["jaccard_top"] == 0.0 and r["spearman"] == pytest.approx(-1.0)


# ---------------------------------------------------------------- A/B
def test_sample_size_matches_reference():
    # Classic reference: p1=0.10, p2=0.15, α=.05, power=.8 → ~686 per arm
    n = abtest.sample_size_two_proportions(0.10, 0.15)
    assert 680 <= n <= 690
    assert abtest.power_two_proportions(0.10, 0.15, n) == pytest.approx(0.80, abs=0.01)


def test_simulated_power_close_to_analytic():
    n = abtest.sample_size_two_proportions(0.5, 0.4)
    assert abtest.simulate(0.5, 0.4, n, n_sims=800, seed=3) == pytest.approx(0.80, abs=0.05)


def test_analyze_detects_effect_and_guardrail():
    rng = np.random.default_rng(0)
    r = abtest.analyze(rng.random(5000) < 0.5, rng.random(5000) < 0.4)
    assert r.p_value < 1e-6 and r.ci_high < 0
    g = abtest.guardrail_check(rng.normal(70, 20, 3000), rng.normal(69.5, 20, 3000), 0.05)
    assert g["passes"]
