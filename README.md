# Telco Customer Churn: Prediction and Retention Profit

Predicts which telecom customers are about to cancel, and, more importantly, **decides who should get a retention offer so the campaign makes money**. Four models are compared fairly, and the decision threshold is set by expected profit, not by accuracy. A business report is generated automatically from each run.

**Skills shown:** scikit-learn pipelines, cross-validated model comparison, metrics for imbalanced data (PR-AUC), profit-based threshold tuning, model explainability (permutation importance, odds ratios), leakage-free evaluation, reproducible reporting, unit tests.

## Results

> Run `python -m src.train`, then copy the headline numbers from `reports/REPORT.md` here. Example format:
> - **Logistic regression beat random forest and gradient boosting** (PR-AUC X vs Y): the simplest model won, and it's also the easiest to explain to a business team.
> - **Targeting customers with a churn score of at least T reaches X% of churners with N offers, for an estimated $Y net**, vs $Z from offering to random customers.
> - **Month-to-month contracts and the first 6 months are the danger zone**: X% and Y% churn.

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
