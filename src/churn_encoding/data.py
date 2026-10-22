"""Loading and cleaning the IBM Telco Customer Churn dataset (paper §3.1)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_PATH = Path("data/raw/Telco-Customer-Churn.csv")

ID_COL = "customerID"
TARGET_COL = "Churn"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is stored as 0/1 but is semantically categorical.
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    ids: pd.Series

    @property
    def churn_rate(self) -> float:
        return float(self.y.mean())


def download(path: Path = DEFAULT_PATH, url: str = DATA_URL) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the simple cleaning rules described in the paper.

    * ``TotalCharges`` is blank for brand-new customers (tenure == 0);
      those blanks become 0.0 rather than being dropped.
    * ``SeniorCitizen`` is cast to string so it is treated as a category.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True))
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype(str)
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype(str)
    return df


def load(path: Path | str = DEFAULT_PATH, auto_download: bool = True) -> Dataset:
    path = Path(path)
    if auto_download:
        download(path)
    df = clean(pd.read_csv(path))
    y = (df[TARGET_COL] == "Yes").astype(int).to_numpy()
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].reset_index(drop=True)
    return Dataset(X=X, y=y, ids=df[ID_COL].reset_index(drop=True))
