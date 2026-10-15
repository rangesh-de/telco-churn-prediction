# Telco Customer Churn: Prediction and Retention Profit

Predicts which telecom customers are about to cancel, and, more importantly, **decides who should get a retention offer so the campaign makes money**. Four models are compared fairly, and the decision threshold is set by expected profit, not by accuracy. A business report is generated automatically from each run.

**Skills shown:** scikit-learn pipelines, cross-validated model comparison, metrics for imbalanced data (PR-AUC), profit-based threshold tuning, model explainability (permutation importance, odds ratios), leakage-free evaluation, reproducible reporting, unit tests.

## Results

On 7,043 customers (26.5% churned), with a 1,409-customer test set held out until the end:

- **Targeting by the model earns 4x more than targeting at random.** Offering a retention deal to customers scoring 0.46 or higher means 620 offers, reaching 308 of 374 churners (82%), for an estimated **$52,157 net**. The same 620 offers sent to random customers net about $12,289, and offering to everyone nets $27,524.
- **The simplest model won.** Logistic regression scored PR-AUC 0.663 in cross-validation, ahead of gradient boosting (0.657) and random forest (0.656). On the test set it scored PR-AUC 0.651 and ROC-AUC 0.845, against 0.27 and 0.50 for no skill. It's also the easiest model to explain to a retention team.
- **Contract type is the biggest churn signal.** 42.7% of month-to-month customers left vs 2.8% on two-year contracts. That makes contract upgrades the obvious offer to test.
- **The first six months are the danger zone.** 52.9% of customers in their first six months churned, vs 9.5% after four years, so onboarding is where retention effort pays most.
- **Fiber customers churn at twice the DSL rate** (41.9% vs 19.0%), which points to a price or service-quality issue worth investigating.

_Estimates assume a $50 offer that keeps 30% of would-be churners, valued at 12 months of charges. Change them with `--offer-cost`, `--save-rate`, `--months`._

Full write-up with charts: **[reports/REPORT.md](reports/REPORT.md)**

## Approach

1. **Clean.** `TotalCharges` is stored as text with blanks for brand-new customers. The blanks are 0 (not billed yet), not missing data, so they're set to 0 instead of dropped or averaged.
2. **Features.** Number of add-on services, an auto-pay flag, and tenure bands.
3. **Hold out 20% as a test set** that is never used for any decision.
4. **Compare 4 models** with 5-fold cross-validation: a no-skill baseline, logistic regression, random forest and gradient boosting, all inside scikit-learn pipelines so preprocessing is learned only from training folds.
5. **Pick the model by PR-AUC.** Only about 1 in 4 customers churns, so accuracy is misleading: predicting "nobody leaves" is already about 74% accurate.
6. **Pick the threshold by profit**, using cross-validated training predictions: offer cost, save rate and customer value are explicit, adjustable assumptions.
7. **Evaluate once on the test set** against two benchmarks: offering to everyone, and offering to the same number of random customers.
8. **Explain:** permutation importance and logistic regression odds ratios.
9. **Act:** a ranked list of at-risk customers by revenue at risk ("who to call first").

## Business assumptions

| Assumption | Default | Flag |
|---|---|---|
| Cost of one retention offer | $50 | `--offer-cost` |
| Share of would-be churners the offer keeps | 30% | `--save-rate` |
| Months of revenue a saved customer is worth | 12 | `--months` |

Example: `python -m src.train --offer-cost 30 --save-rate 0.2`. Changing these moves the best threshold, which is the point: the model's job depends on the business's economics.

## How to run it

```bash
pip install -r requirements.txt
# Download the CSV from Kaggle into data/ (see data/README.md)
python -m src.train        # about 30 seconds
python -m pytest
```

Outputs:
- `reports/REPORT.md`: results and business impact, with charts
- `reports/model_comparison.csv`: cross-validation scores
- `reports/metrics.json`: every number from the run
- `reports/top_at_risk_customers.csv`: who to call first
- `models/churn_model.joblib`: the trained pipeline

## Project structure

```
├── data/                 Raw Kaggle CSV (not committed)
├── docs/WALKTHROUGH.md   How each step works + interview questions
├── reports/              Generated report, figures, metrics
├── src/
│   ├── data.py           Validation, cleaning, features
│   ├── business.py       Campaign profit and threshold choice
│   ├── train.py          Model comparison, evaluation, explainability
│   └── report.py         Writes REPORT.md from the run's numbers
└── tests/test_churn.py
```

## Limitations

- One snapshot with no dates: the model ranks risk, it doesn't predict *when* someone leaves.
- Profit depends on assumed offer economics; a real rollout would A/B test the offer to measure the true save rate.
- IBM's Telco dataset is a teaching sample, so results show method, not a real company's numbers.

## Data source

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data) on Kaggle.
