"""Load, validate, clean and engineer features for the IBM Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    *ADDON_SERVICES, "Contract", "PaperlessBilling", "PaymentMethod", "TenureBand",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddons", "AutoPay"]
TARGET = "Churned"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Is this the Telco Customer Churn file?")
    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values")
    if not set(df["Churn"].unique()) <= {"Yes", "No"}:
        raise ValueError("Churn should only contain Yes/No")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    log = {}

    # TotalCharges is text in the source; customers with tenure 0 have a blank because they
    # haven't been billed yet. Their correct total is 0, not "missing".
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = df["TotalCharges"].isna()
    log["blank_total_charges"] = int(blank.sum())
    log["blank_total_charges_with_tenure_0"] = int((blank & (df["tenure"] == 0)).sum())
    df.loc[blank & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna(df["SeniorCitizen"].astype(str))
    df[TARGET] = (df["Churn"] == "Yes").astype(int)

    # Features
    df["NumAddons"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)
    df["AutoPay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    df["TenureBand"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 100],
                              labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]).astype(str)

    log["customers"] = len(df)
    log["churn_rate"] = round(float(df[TARGET].mean()), 4)
    return df, log


def features_and_target(df: pd.DataFrame):
    return df[CATEGORICAL + NUMERIC], df[TARGET]
