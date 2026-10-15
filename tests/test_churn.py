import numpy as np
import pandas as pd
import pytest

from src.business import Assumptions, best_threshold, campaign_profit
from src.data import clean, features_and_target, validate
from src.train import models


def raw(n=200, seed=0):
    rng = np.random.default_rng(seed)
    yn = lambda: rng.choice(["Yes", "No"], n)
    tenure = rng.integers(1, 72, n)
    tenure[:3] = 0
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    df = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)], "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n), "Partner": yn(), "Dependents": yn(), "tenure": tenure,
        "PhoneService": yn(), "MultipleLines": yn(), "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": yn(), "OnlineBackup": yn(), "DeviceProtection": yn(), "TechSupport": yn(),
        "StreamingTV": yn(), "StreamingMovies": yn(), "Contract": contract, "PaperlessBilling": yn(),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Credit card (automatic)"], n),
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
    })
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    df["Churn"] = np.where((contract == "Month-to-month") & (rng.random(n) < 0.7), "Yes", "No")
    return df


def test_validate_rejects_wrong_file_and_duplicates():
    with pytest.raises(ValueError, match="Missing columns"):
        validate(pd.DataFrame({"x": [1]}))
    df = raw()
    df.loc[1, "customerID"] = "C0"
    with pytest.raises(ValueError, match="Duplicate"):
        validate(df)


def test_blank_total_charges_become_zero_for_new_customers():
    df, log = clean(raw())
    assert log["blank_total_charges"] == 3 == log["blank_total_charges_with_tenure_0"]
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert df["TotalCharges"].notna().all()


def test_engineered_features():
    df, _ = clean(raw())
    assert set(df["SeniorCitizen"]) <= {"Yes", "No"}
    assert df["NumAddons"].between(0, 6).all()
    assert df.loc[df["PaymentMethod"] == "Credit card (automatic)", "AutoPay"].eq(1).all()
    assert df.loc[df["PaymentMethod"] == "Mailed check", "AutoPay"].eq(0).all()
    assert (df["TenureBand"] != "nan").all()


def test_campaign_profit_arithmetic():
    a = Assumptions(offer_cost=10, save_rate=0.5, months_retained=2)
    y = [1, 1, 0, 0]
    scores = [0.9, 0.2, 0.8, 0.1]
    monthly = [100, 100, 100, 100]
    r = campaign_profit(y, scores, monthly, 0.5, a)
    # targets customers 0 and 2; saves 0.5 x 100 x 2 = 100 from the one churner; costs 2 x 10
    assert r["customers_targeted"] == 2 and r["churners_reached"] == 1
    assert r["value_saved"] == 100 and r["campaign_cost"] == 20 and r["net_profit"] == 80


def test_expensive_offers_push_threshold_up():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    scores = np.clip(y * 0.4 + rng.uniform(0, 0.6, 500), 0, 1)
    monthly = np.full(500, 70.0)
    cheap = best_threshold(y, scores, monthly, Assumptions(offer_cost=5))["threshold"]
    pricey = best_threshold(y, scores, monthly, Assumptions(offer_cost=200))["threshold"]
    assert pricey > cheap


def test_every_model_trains_and_learns_signal():
    df, _ = clean(raw(400))
    X, y = features_and_target(df)
    for name, pipe in models().items():
        p = pipe.fit(X, y).predict_proba(X)[:, 1]
        assert p.shape == (400,) and ((p >= 0) & (p <= 1)).all(), name
    lr = models()["Logistic regression"].fit(X, y)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, lr.predict_proba(X)[:, 1]) > 0.7
