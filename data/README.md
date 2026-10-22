# Data

`raw/Telco-Customer-Churn.csv` is the public **IBM Telco Customer Churn** sample
(7,043 customers, 21 columns), mirrored from
[IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data).
It is a fictional dataset published by IBM for demonstrations.

If the file is missing, `churn_encoding.data.load()` downloads it automatically
(or run `make data`).

Cleaning rules (`src/churn_encoding/data.py`):

* 11 customers with `tenure == 0` have a blank `TotalCharges` → set to `0.0`.
* `SeniorCitizen` (0/1) is treated as a categorical `No`/`Yes`.
* `customerID` is kept only as an identifier, never as a feature.
