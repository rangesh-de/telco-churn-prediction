"""
End-to-end churn modelling run.

    python -m src.train                      # uses data/WA_Fn-UseC_-Telco-Customer-Churn.csv
    python -m src.train --offer-cost 40 --save-rate 0.25 --months 12

Steps:
  1. Clean data and save exploratory charts.
  2. Hold out 20% as a test set that is never used for any decision.
  3. Compare 4 models with 5-fold cross-validation on the training set.
  4. Pick the best model by PR-AUC, then pick the decision threshold that maximizes
     campaign profit using out-of-fold predictions (still no test data involved).
  5. Evaluate once on the test set, explain the model, and write the business report.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix, f1_score, make_scorer,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.business import Assumptions, best_threshold, campaign_profit, profit_curve  # noqa: E402
from src.data import CATEGORICAL, NUMERIC, clean, features_and_target, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 42
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"


def preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ("num", numeric, NUMERIC),
    ])


def models() -> dict[str, Pipeline]:
    return {
        "Baseline (no skill)": Pipeline([("prep", preprocessor(False)),
                                         ("model", DummyClassifier(strategy="prior"))]),
        "Logistic regression": Pipeline([("prep", preprocessor(True)),
                                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random forest": Pipeline([("prep", preprocessor(False)),
                                   ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                                    class_weight="balanced", n_jobs=-1,
                                                                    random_state=SEED))]),
        "Gradient boosting": Pipeline([("prep", preprocessor(False)),
                                       ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                                max_leaf_nodes=15,
                                                                                class_weight="balanced",
                                                                                random_state=SEED))]),
    }


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


def eda_charts(df: pd.DataFrame) -> dict:
    overall = df["Churned"].mean()
    rates = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col, order in zip(
        axes.flat,
        ["Contract", "TenureBand", "InternetService", "PaymentMethod"],
        [["Month-to-month", "One year", "Two year"], ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"],
         None, None],
    ):
        r = df.groupby(col)["Churned"].mean().mul(100)
        r = r.reindex(order) if order else r.sort_values(ascending=False)
        rates[col] = r.round(1).to_dict()
        ax.barh(r.index.astype(str), r.values, color=[ORANGE if v > overall * 100 else BLUE for v in r.values])
        ax.axvline(overall * 100, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"Churn rate by {col} (%)")
        ax.invert_yaxis()
        for y, v in enumerate(r.values):
            ax.text(v + 0.5, y, f"{v:.0f}%", va="center", fontsize=9)
    fig.suptitle(f"Who churns? Dashed line = overall churn rate ({overall:.1%})", y=1.01)
    fig.tight_layout()
    save(fig, "01_churn_by_segment")
    return rates


def compare_models(X_train, y_train) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"ROC-AUC": "roc_auc", "PR-AUC": "average_precision", "Recall": "recall",
               "Precision": make_scorer(precision_score, zero_division=0), "F1": "f1"}
    rows = []
    for name, pipe in models().items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            vals = res[f"test_{metric}"]
            row[metric] = round(float(vals.mean()), 3)
            row[f"{metric} sd"] = round(float(vals.std()), 3)
        rows.append(row)
        print(f"  {name:<22} ROC-AUC {row['ROC-AUC']:.3f}  PR-AUC {row['PR-AUC']:.3f}  Recall {row['Recall']:.3f}")
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)


def test_curves(fitted: dict, X_test, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [GREY, BLUE, "#38a169", ORANGE]
    for (name, pipe), c in zip(fitted.items(), colors):
        p = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax1.plot(fpr, tpr, color=c, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
        ax2.plot(rec, prec, color=c, label=f"{name} ({average_precision_score(y_test, p):.3f})")
    ax1.plot([0, 1], [0, 1], color="black", linewidth=0.6, linestyle=":")
    ax1.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax2.set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    ax1.legend(fontsize=8, title="AUC")
    ax2.legend(fontsize=8, title="Average precision")
    save(fig, "02_roc_pr_curves")


def profit_chart(curve: list[dict], chosen: float, a: Assumptions):
    c = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(c["threshold"], c["net_profit"], color=BLUE, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(chosen, color=ORANGE, linestyle="--")
    best = c.loc[c["threshold"] == chosen].iloc[0]
    ax.annotate(f"Chosen threshold {chosen:.2f}\n${best['net_profit']:,.0f} net, {int(best['customers_targeted']):,} offers",
                xy=(chosen, best["net_profit"]), xytext=(10, -40), textcoords="offset points", fontsize=9)
    ax.axvline(0.5, color=GREY, linestyle=":")
    ax.text(0.505, ax.get_ylim()[0], " default 0.5", color="grey", fontsize=8, va="bottom")
    ax.set(title=f"Campaign profit by risk threshold (test set)\nOffer ${a.offer_cost:.0f}, "
                 f"save rate {a.save_rate:.0%}, value = {a.months_retained} months of charges",
           xlabel="Target customers with churn score at or above", ylabel="Net profit ($)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    save(fig, "03_profit_curve")


def explain(best_pipe, X_test, y_test, fitted: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    perm = permutation_importance(best_pipe, X_test, y_test, scoring="average_precision",
                                  n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean,
                         "sd": perm.importances_std})
           .sort_values("importance", ascending=False).head(12))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], xerr=imp["sd"][::-1], color=BLUE)
    ax.set(title="What the model relies on (permutation importance, test set)",
           xlabel="Drop in PR-AUC when the feature is shuffled")
    save(fig, "04_feature_importance")

    # Odds ratios from the logistic regression: direction and size, in plain terms
    lr = fitted["Logistic regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = lr.named_steps["model"].coef_[0]
    odds = (pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "odds_ratio": np.exp(coefs)})
            .assign(effect=lambda d: np.where(d["odds_ratio"] >= 1, "raises churn odds", "lowers churn odds"))
            .sort_values("odds_ratio", key=lambda s: (np.log(s)).abs(), ascending=False).head(12))
    odds["odds_ratio"] = odds["odds_ratio"].round(2)
    return imp, odds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--offer-cost", type=float, default=50.0)
    parser.add_argument("--save-rate", type=float, default=0.30)
    parser.add_argument("--months", type=int, default=12)
    args = parser.parse_args()
    a = Assumptions(args.offer_cost, args.save_rate, args.months)

    if not args.data.exists():
        raise SystemExit(f"Data not found at {args.data}. See data/README.md.")
    df, log = clean(load(args.data))
    print(f"{log['customers']:,} customers, churn rate {log['churn_rate']:.1%}")
    rates = eda_charts(df)

    X, y = features_and_target(df)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, stratify=y, random_state=SEED)

    print("\n5-fold cross-validation on the training set")
    comparison = compare_models(X_train, y_train)
    best_name = comparison.iloc[0]["model"]
    print(f"Best by PR-AUC: {best_name}")

    # Threshold chosen on out-of-fold predictions: the test set plays no part in any decision
    oof = cross_val_predict(models()[best_name], X_train, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    chosen = best_threshold(y_train, oof, df_train["MonthlyCharges"], a)["threshold"]

    fitted = {name: pipe.fit(X_train, y_train) for name, pipe in models().items()}
    best = fitted[best_name]
    p_test = best.predict_proba(X_test)[:, 1]

    test_metrics = {}
    for label, t in [("default_0.5", 0.5), ("profit_threshold", chosen)]:
        pred = (p_test >= t).astype(int)
        test_metrics[label] = {
            "threshold": t, "recall": round(recall_score(y_test, pred), 3),
            "precision": round(precision_score(y_test, pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, pred), 3),
            **campaign_profit(y_test, p_test, df_test["MonthlyCharges"], t, a),
        }
    test_metrics["roc_auc"] = round(roc_auc_score(y_test, p_test), 3)
    test_metrics["pr_auc"] = round(average_precision_score(y_test, p_test), 3)
    test_metrics["target_everyone"] = campaign_profit(y_test, np.ones_like(p_test), df_test["MonthlyCharges"], 0, a)

    # Random targeting of the same number of customers, averaged over 200 draws
    rng = np.random.default_rng(SEED)
    n_target = test_metrics["profit_threshold"]["customers_targeted"]
    random_profits = [campaign_profit(y_test, rng.permutation(len(p_test)) < n_target,
                                      df_test["MonthlyCharges"], 0.5, a)["net_profit"] for _ in range(200)]
    test_metrics["random_same_size_net_profit"] = round(float(np.mean(random_profits)), 2)

    test_curves(fitted, X_test, y_test)
    curve = profit_curve(y_test, p_test, df_test["MonthlyCharges"], a)
    profit_chart(curve, chosen, a)
    cm = confusion_matrix(y_test, (p_test >= chosen).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test set at threshold {chosen:.2f}")
    save(fig, "05_confusion_matrix")
    importance, odds = explain(best, X_test, y_test, fitted)

    # Who to call first: highest expected revenue at risk
    at_risk = df_test[["customerID", "Contract", "tenure", "MonthlyCharges"]].copy()
    at_risk["churn_score"] = p_test.round(3)
    at_risk["revenue_at_risk"] = (at_risk["churn_score"] * at_risk["MonthlyCharges"] * a.months_retained).round(2)
    at_risk = at_risk[at_risk["churn_score"] >= chosen].sort_values("revenue_at_risk", ascending=False)

    REPORTS.mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models" / "churn_model.joblib")
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    at_risk.head(50).to_csv(REPORTS / "top_at_risk_customers.csv", index=False)
    results = {"data": log, "best_model": best_name, "assumptions": a.__dict__, "chosen_threshold": chosen,
               "test": test_metrics, "churn_rate_by_segment": rates}
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

    from src.report import write_report
    write_report(results, comparison, importance, odds, at_risk.head(10), REPORTS / "REPORT.md")
    print(f"\nTest PR-AUC {test_metrics['pr_auc']}, ROC-AUC {test_metrics['roc_auc']}")
    print(f"Profit at 0.5: ${test_metrics['default_0.5']['net_profit']:,.0f}   "
          f"at {chosen}: ${test_metrics['profit_threshold']['net_profit']:,.0f}   "
          f"random same size: ${test_metrics['random_same_size_net_profit']:,.0f}")
    print(f"Report: {REPORTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
