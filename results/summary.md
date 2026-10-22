# Results summary (auto-generated)

Dataset: 7,043 customers, churn rate 26.54%. Protocol: stratified 5-fold (3 train / 1 val / 1 test), seeds [0, 1, 2] → 15 test folds per cell.

## RQ1 — Predictive performance (mean ± sd over test folds)

| model | encoder | auc | f1 | precision | recall | brier | #features |
|---|---|---|---|---|---|---|---|
| gbt | onehot | 0.8394 ± 0.0145 | 0.6221 ± 0.0196 | 0.5434 ± 0.0295 | 0.7343 ± 0.0642 | 0.1582 ± 0.0059 | 46 |
| gbt | target | 0.8430 ± 0.0127 | 0.6215 ± 0.0216 | 0.5550 ± 0.0295 | 0.7141 ± 0.0730 | 0.1580 ± 0.0055 | 19 |
| logreg | onehot | 0.8438 ± 0.0115 | 0.6251 ± 0.0192 | 0.5548 ± 0.0289 | 0.7236 ± 0.0700 | 0.1658 ± 0.0060 | 46 |
| logreg | target | 0.8439 ± 0.0116 | 0.6245 ± 0.0192 | 0.5546 ± 0.0315 | 0.7245 ± 0.0774 | 0.1655 ± 0.0058 | 19 |

Paired differences (target − one-hot, same folds):

| model | metric | mean_diff_target_minus_onehot | wins_target | n_folds | p_value_paired_t |
|---|---|---|---|---|---|
| logreg | auc | 0.0000 | 8 | 15 | 0.9072 |
| logreg | f1 | -0.0006 | 6 | 15 | 0.8529 |
| logreg | brier | -0.0003 | 6 | 15 | 0.2546 |
| gbt | auc | 0.0035 | 12 | 15 | 0.0078 |
| gbt | f1 | -0.0006 | 7 | 15 | 0.7913 |
| gbt | brier | -0.0002 | 6 | 15 | 0.6846 |

## RQ2 — Targeting stability (top 10%, averaged over seeds)

| a | b | spearman | jaccard_top | overlap_share |
|---|---|---|---|---|
| GBT+One-hot | GBT+Target | 0.9743 | 0.7139 | 0.8326 |
| LogReg+One-hot | GBT+One-hot | 0.9406 | 0.5583 | 0.7163 |
| LogReg+One-hot | GBT+Target | 0.9503 | 0.5820 | 0.7357 |
| LogReg+One-hot | LogReg+Target | 0.9939 | 0.8668 | 0.9286 |
| LogReg+Target | GBT+One-hot | 0.9373 | 0.5452 | 0.7054 |
| LogReg+Target | GBT+Target | 0.9499 | 0.5697 | 0.7258 |

Seed-noise baseline (same encoder + model, different CV split):

| cell | spearman | jaccard_top |
|---|---|---|
| GBT+One-hot | 0.9460 | 0.6013 |
| GBT+Target | 0.9538 | 0.6358 |
| LogReg+One-hot | 0.9915 | 0.8108 |
| LogReg+Target | 0.9944 | 0.8009 |

Who is in each targeted group (seed 0):

| group | n | actual_churn_rate | mean_tenure | mean_monthly_charges | senior_share | contract=Month-to-month | contract=Two year | contract=One year |
|---|---|---|---|---|---|---|---|---|
| all customers | 7043 | 0.2654 | 32.3711 | 64.7617 | 0.1621 | 0.5502 | 0.2407 | 0.2091 |
| LogReg top10 one-hot | 705 | 0.7574 | 7.6369 | 83.9905 | 0.4113 | 1.0000 | 0.0000 | 0.0000 |
| LogReg top10 target | 705 | 0.7546 | 7.1787 | 83.4884 | 0.4028 | 1.0000 | 0.0000 | 0.0000 |
| LogReg one-hot only | 45 | 0.6667 | 14.2222 | 83.6378 | 0.3111 | 1.0000 | 0.0000 | 0.0000 |
| LogReg target only | 45 | 0.6222 | 7.0444 | 75.7711 | 0.1778 | 1.0000 | 0.0000 | 0.0000 |
| GBT top10 one-hot | 705 | 0.7504 | 5.7518 | 80.0605 | 0.3121 | 1.0000 | 0.0000 | 0.0000 |
| GBT top10 target | 705 | 0.7560 | 5.2766 | 80.8443 | 0.3149 | 1.0000 | 0.0000 | 0.0000 |
| GBT one-hot only | 108 | 0.6111 | 10.7130 | 73.7542 | 0.2130 | 1.0000 | 0.0000 | 0.0000 |
| GBT target only | 108 | 0.6481 | 7.6111 | 78.8704 | 0.2315 | 1.0000 | 0.0000 | 0.0000 |

## RQ3 — A/B test sizing

Chosen model (best mean AUC): **LogReg + Target**. Eligibility pool = top 10% risk; observed baseline churn in that pool = **75.46%**.

| p_control | relative_mde | p_treatment | absolute_mde | n_per_arm | n_total | months_to_enroll |
|---|---|---|---|---|---|---|
| 0.7546 | 0.0500 | 0.7169 | 0.0377 | 2143.0000 | 4286.0000 | 6.1229 |
| 0.7546 | 0.1000 | 0.6791 | 0.0755 | 559.0000 | 1118.0000 | 1.5971 |
| 0.7546 | 0.1500 | 0.6414 | 0.1132 | 258.0000 | 516.0000 | 0.7371 |
| 0.7546 | 0.2000 | 0.6037 | 0.1509 | 149.0000 | 298.0000 | 0.4257 |
| 0.7546 | 0.2500 | 0.5660 | 0.1887 | 98.0000 | 196.0000 | 0.2800 |
| 0.7546 | 0.3000 | 0.5282 | 0.2264 | 70.0000 | 140.0000 | 0.2000 |

Monte-Carlo check: simulated power at 20% relative MDE = **0.797** (target 0.8).
