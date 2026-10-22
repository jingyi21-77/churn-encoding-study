#!/usr/bin/env python
"""Run the full study: RQ1 (performance), RQ2 (targeting stability), RQ3 (A/B design).

    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --quick        # 1 seed, small grid (smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn_encoding import abtest, plots, stability  # noqa: E402
from churn_encoding.cv import run_cell  # noqa: E402
from churn_encoding.data import load  # noqa: E402
from churn_encoding.plots import LABEL  # noqa: E402

METRICS = ["auc", "f1", "precision", "recall", "brier", "log_loss"]


def key_name(enc, mod):
    return f"{LABEL[mod]}+{LABEL[enc]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--quick", action="store_true", help="1 seed + tiny grid, for CI/smoke tests")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.quick:
        cfg["cv"]["seeds"] = [0]
        cfg["search_space"] = {"logreg": {"C": [1.0]},
                               "gbt": {"max_depth": [3], "learning_rate": [0.1],
                                       "subsample": [0.8], "n_estimators": [150]}}
    out = Path(args.output_dir or cfg["output_dir"])
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ds = load(cfg["data_path"])
    print(f"Loaded {len(ds.y):,} customers, churn rate {ds.churn_rate:.3%}, "
          f"{ds.X.shape[1]} raw features")

    # ------------------------------------------------------------------ RQ1
    t0 = time.time()
    cells, fold_rows = {}, []
    enc_kw = {"smooth": cfg["target_encoder"]["smooth"], "inner_cv": cfg["target_encoder"]["inner_cv"]}
    for seed in cfg["cv"]["seeds"]:
        for mod in cfg["models"]:
            for enc in cfg["encoders"]:
                r = run_cell(ds.X, ds.y, enc, mod, cfg["search_space"][mod],
                             n_splits=cfg["cv"]["n_splits"], seed=seed,
                             enc_kw=enc_kw if enc == "target" else {})
                cells[(enc, mod, seed)] = r
                fold_rows += r.fold_rows
    print(f"CV finished in {time.time() - t0:.0f}s")

    folds = pd.DataFrame(fold_rows)
    folds["params"] = folds["params"].map(json.dumps)
    folds.to_csv(out / "fold_metrics.csv", index=False)

    summ = (folds.groupby(["model", "encoder"])[METRICS + ["n_features"]]
            .agg(["mean", "std"]).round(4))
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "summary_metrics.csv", index=False)

    # paired comparison target − onehot on identical folds
    paired = []
    for mod in cfg["models"]:
        a = folds[(folds.model == mod) & (folds.encoder == "onehot")].sort_values(["seed", "fold"])
        b = folds[(folds.model == mod) & (folds.encoder == "target")].sort_values(["seed", "fold"])
        for m in METRICS:
            d = b[m].to_numpy() - a[m].to_numpy()
            t = ttest_rel(b[m], a[m])
            paired.append({"model": mod, "metric": m, "mean_diff_target_minus_onehot": d.mean(),
                           "sd_diff": d.std(ddof=1), "n_folds": len(d),
                           "wins_target": int((d > 0).sum()), "p_value_paired_t": t.pvalue})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired_differences.csv", index=False)

    # ------------------------------------------------------------------ RQ2
    q = cfg["stability"]["top_q"]
    stab_rows = []
    for seed in cfg["cv"]["seeds"]:
        sc = {key_name(e, m): cells[(e, m, seed)].oof_score
              for m in cfg["models"] for e in cfg["encoders"]}
        t = stability.pairwise_table(sc, q)
        t["seed"] = seed
        t["comparison"] = "encoder/model"
        stab_rows.append(t)
    # noise baseline: same encoder + model, different fold split (seed)
    seeds = cfg["cv"]["seeds"]
    if len(seeds) > 1:
        for m in cfg["models"]:
            for e in cfg["encoders"]:
                for i, s1 in enumerate(seeds):
                    for s2 in seeds[i + 1:]:
                        c = stability.compare(cells[(e, m, s1)].oof_score, cells[(e, m, s2)].oof_score, q)
                        stab_rows.append(pd.DataFrame([{"a": f"{key_name(e, m)}@seed{s1}",
                                                        "b": f"{key_name(e, m)}@seed{s2}",
                                                        **c, "seed": -1,
                                                        "comparison": "seed-noise baseline"}]))
    stab = pd.concat(stab_rows, ignore_index=True)
    stab.to_csv(out / "stability_all.csv", index=False)

    enc_pairs = stab[(stab.comparison == "encoder/model")]
    stab_summary = (enc_pairs.groupby(["a", "b"])[["spearman", "jaccard_top", "overlap_share",
                                                   "n_only_first", "n_only_second"]]
                    .mean().round(4).reset_index())
    stab_summary.to_csv(out / "stability_summary.csv", index=False)
    noise = stab[stab.comparison == "seed-noise baseline"].copy()
    noise_summary = None
    if len(noise):
        noise["cell"] = noise["a"].str.split("@").str[0]
        noise_summary = noise.groupby("cell")[["spearman", "jaccard_top"]].mean().round(4).reset_index()
        noise_summary.to_csv(out / "stability_seed_noise.csv", index=False)

    # segment profile (seed 0, GBT and LogReg)
    s0 = seeds[0]
    groups = {"all customers": np.ones(len(ds.y), bool)}
    for m in cfg["models"]:
        a = stability.top_q_mask(cells[("onehot", m, s0)].oof_score, q)
        b = stability.top_q_mask(cells[("target", m, s0)].oof_score, q)
        groups.update({f"{LABEL[m]} top10 one-hot": a, f"{LABEL[m]} top10 target": b,
                       f"{LABEL[m]} one-hot only": a & ~b, f"{LABEL[m]} target only": b & ~a})
    prof = stability.segment_profile(ds.X, ds.y, groups)
    prof.round(4).to_csv(out / "segment_profile.csv", index=False)

    # pooled OOF scores (seed 0) for downstream use
    oof = pd.DataFrame({"customerID": ds.ids, "churn": ds.y})
    for (e, m, s), r in cells.items():
        oof[f"{m}_{e}_seed{s}"] = r.oof_score
    oof.to_csv(out / "oof_scores.csv.gz", index=False, compression="gzip")

    # ------------------------------------------------------------------ RQ3
    best = summ.sort_values("auc_mean", ascending=False).iloc[0]
    best_key = (best["encoder"], best["model"])
    elig = stability.top_q_mask(cells[(best_key[0], best_key[1], s0)].oof_score,
                                cfg["abtest"]["eligible_top_q"])
    p0 = float(ds.y[elig].mean())
    ab_rows = []
    for rel in cfg["abtest"]["relative_mde"]:
        p1 = p0 * (1 - rel)
        n = abtest.sample_size_two_proportions(p0, p1, cfg["abtest"]["alpha"], cfg["abtest"]["power"])
        ab_rows.append({"p_control": p0, "relative_mde": rel, "p_treatment": p1,
                        "absolute_mde": p0 - p1, "n_per_arm": n, "n_total": 2 * n,
                        "months_to_enroll": 2 * n / cfg["abtest"]["monthly_eligible_customers"]})
    ab = pd.DataFrame(ab_rows)
    sim_rel = 0.20 if 0.20 in cfg["abtest"]["relative_mde"] else cfg["abtest"]["relative_mde"][-1]
    sim_row = ab[ab.relative_mde == sim_rel].iloc[0]
    sim_power = abtest.simulate(p0, sim_row["p_treatment"], int(sim_row["n_per_arm"]),
                                n_sims=500 if args.quick else 2000)
    ab.round(4).to_csv(out / "ab_sample_size.csv", index=False)

    # ------------------------------------------------------------------ figures
    sc0 = {(e, m): cells[(e, m, s0)].oof_score for m in cfg["models"] for e in cfg["encoders"]}
    plots.roc(sc0, ds.y, fig_dir / "roc.png")
    plots.calibration(sc0, ds.y, fig_dir / "calibration.png")
    plots.score_scatter(sc0, fig_dir / "score_scatter_top10.png", q)
    labels = [key_name(e, m) for m in cfg["models"] for e in cfg["encoders"]]
    plots.stability_heatmap(stab_summary, labels, "jaccard_top", fig_dir / "jaccard_top10.png",
                            f"Jaccard overlap of top-{int(q*100)}% lists")
    plots.stability_heatmap(stab_summary, labels, "spearman", fig_dir / "spearman.png",
                            "Spearman correlation of risk scores")
    plots.power_curve(ab, fig_dir / "ab_power.png")

    # ------------------------------------------------------------------ report
    write_report(out, cfg, ds, summ, paired, stab_summary, noise_summary, prof, ab, best_key, p0,
                 sim_rel, sim_power)
    print(f"\nDone. See {out}/summary.md")


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out, cfg, ds, summ, paired, stab_summary, noise_summary, prof, ab, best_key, p0,
                 sim_rel, sim_power):
    n_runs = len(cfg["cv"]["seeds"]) * cfg["cv"]["n_splits"]
    t = summ.copy()
    for m in ["auc", "f1", "precision", "recall", "brier"]:
        t[m] = t[f"{m}_mean"].map("{:.4f}".format) + " ± " + t[f"{m}_std"].map("{:.4f}".format)
    t["#features"] = t["n_features_mean"].astype(int)
    t = t[["model", "encoder", "auc", "f1", "precision", "recall", "brier", "#features"]]

    p = paired[paired.metric.isin(["auc", "f1", "brier"])].copy()
    p = p[["model", "metric", "mean_diff_target_minus_onehot", "wins_target", "n_folds", "p_value_paired_t"]]

    s = stab_summary[["a", "b", "spearman", "jaccard_top", "overlap_share"]]
    prof_cols = ["group", "n", "actual_churn_rate", "mean_tenure", "mean_monthly_charges",
                 "senior_share"] + [c for c in prof.columns if c.startswith("contract=")]
    lines = [
        "# Results summary (auto-generated)",
        "",
        f"Dataset: {len(ds.y):,} customers, churn rate {ds.churn_rate:.2%}. "
        f"Protocol: stratified {cfg['cv']['n_splits']}-fold (3 train / 1 val / 1 test), "
        f"seeds {cfg['cv']['seeds']} → {n_runs} test folds per cell.",
        "",
        "## RQ1 — Predictive performance (mean ± sd over test folds)",
        "", _md(t), "",
        "Paired differences (target − one-hot, same folds):", "", _md(p), "",
        "## RQ2 — Targeting stability (top "
        f"{int(cfg['stability']['top_q'] * 100)}%, averaged over seeds)",
        "", _md(s), "",
    ]
    if noise_summary is not None:
        lines += ["Seed-noise baseline (same encoder + model, different CV split):", "",
                  _md(noise_summary), ""]
    lines += ["Who is in each targeted group (seed 0):", "", _md(prof[prof_cols]), "",
              "## RQ3 — A/B test sizing",
              "",
              f"Chosen model (best mean AUC): **{LABEL[best_key[1]]} + {LABEL[best_key[0]]}**. "
              f"Eligibility pool = top {int(cfg['abtest']['eligible_top_q'] * 100)}% risk; "
              f"observed baseline churn in that pool = **{p0:.2%}**.",
              "", _md(ab), "",
              f"Monte-Carlo check: simulated power at {int(sim_rel * 100)}% relative MDE = "
              f"**{sim_power:.3f}** (target {cfg['abtest']['power']}).", ""]
    (out / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
