"""A/B test design blueprint (paper §5, RQ3).

* ``sample_size_two_proportions`` — per-arm n for a two-sided test of
  p_control vs p_treatment (normal approximation, pooled variance under H0).
* ``power_two_proportions`` — power for a given per-arm n.
* ``analyze`` — two-proportion z-test + Wald CI for the treatment effect,
  used when the experiment's data come back.
* ``simulate`` — Monte-Carlo check that the planned design hits its power.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import norm


def sample_size_two_proportions(p_control: float, p_treatment: float,
                                alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-arm sample size, equal allocation, two-sided α."""
    if not (0 < p_control < 1 and 0 < p_treatment < 1) or p_control == p_treatment:
        raise ValueError("need 0 < p < 1 and p_control != p_treatment")
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p_bar = (p_control + p_treatment) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * np.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))) ** 2
    return int(np.ceil(num / (p_control - p_treatment) ** 2))


def power_two_proportions(p_control: float, p_treatment: float, n_per_arm: int,
                          alpha: float = 0.05) -> float:
    z_a = norm.ppf(1 - alpha / 2)
    p_bar = (p_control + p_treatment) / 2
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)
    se1 = np.sqrt((p_control * (1 - p_control) + p_treatment * (1 - p_treatment)) / n_per_arm)
    d = abs(p_control - p_treatment)
    return float(norm.cdf((d - z_a * se0) / se1) + norm.cdf((-d - z_a * se0) / se1))


@dataclass
class ABResult:
    n_control: int
    n_treatment: int
    churn_control: float
    churn_treatment: float
    diff: float              # treatment - control (negative = offer reduces churn)
    ci_low: float
    ci_high: float
    z: float
    p_value: float

    def to_dict(self):
        return asdict(self)


def analyze(churn_control: np.ndarray, churn_treatment: np.ndarray, alpha: float = 0.05) -> ABResult:
    c, t = np.asarray(churn_control), np.asarray(churn_treatment)
    n0, n1 = len(c), len(t)
    p0, p1 = c.mean(), t.mean()
    pool = (c.sum() + t.sum()) / (n0 + n1)
    se_pool = np.sqrt(pool * (1 - pool) * (1 / n0 + 1 / n1))
    z = (p1 - p0) / se_pool if se_pool > 0 else 0.0
    se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    zc = norm.ppf(1 - alpha / 2)
    return ABResult(n0, n1, float(p0), float(p1), float(p1 - p0),
                    float(p1 - p0 - zc * se), float(p1 - p0 + zc * se),
                    float(z), float(2 * (1 - norm.cdf(abs(z)))))


def guardrail_check(control: np.ndarray, treatment: np.ndarray, max_relative_drop: float) -> dict:
    """Non-inferiority style guardrail on a continuous metric (e.g. revenue).

    Passes if the lower 95% CI of (treatment/control - 1) is above
    ``-max_relative_drop``.
    """
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    mc, mt = c.mean(), t.mean()
    rel = mt / mc - 1
    # delta-method SE of the ratio
    se = np.sqrt(t.var(ddof=1) / len(t) / mc**2 + mt**2 * c.var(ddof=1) / len(c) / mc**4)
    lo = rel - 1.96 * se
    return {"relative_change": float(rel), "ci_low": float(lo), "passes": bool(lo > -max_relative_drop)}


def simulate(p_control: float, p_treatment: float, n_per_arm: int, n_sims: int = 2000,
             alpha: float = 0.05, seed: int = 0) -> float:
    """Empirical power of the planned design via Monte-Carlo."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        c = rng.random(n_per_arm) < p_control
        t = rng.random(n_per_arm) < p_treatment
        hits += analyze(c, t, alpha).p_value < alpha
    return hits / n_sims
