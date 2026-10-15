# Walkthrough

How each step works, why it was done that way, and what to say in an interview.

## 1. Data cleaning (`src/data.py`)

- `TotalCharges` comes in as text. 11 rows are blank, all with `tenure = 0`. These are new customers who haven't been billed yet, so the correct value is **0**. Dropping them or filling with the average would both be wrong.
- **Interview line:** "Before treating blanks as missing, I checked what they meant. They were all brand-new customers."

## 2. Train/test split

- 80/20, **stratified**, so both sets have the same churn rate.
- The test set is used **once**, at the end. Model choice and threshold choice both happen on training data only. If the test set influences decisions, its score stops being an honest estimate.

## 3. Pipelines

- Each model is a `Pipeline`: one-hot encoding (+ scaling for logistic regression) → model.
- Inside cross-validation, the encoder and scaler are fitted on the training folds only. Scaling the full dataset first would **leak** information from the validation fold.

## 4. Model comparison

- **Baseline (no skill)** predicts the average churn rate for everyone. Every real model must beat it, and it shows what "no skill" looks like on each metric.
- **Logistic regression**: simple, fast, explainable.
- **Random forest / gradient boosting**: capture interactions and non-linear effects.
- `class_weight="balanced"` makes models pay more attention to the minority class (churners).
- **Why PR-AUC:** with ~26% churn, ROC-AUC can look good while the model still produces many false alarms. PR-AUC focuses on how precise the model is while finding churners.
- **If logistic regression wins** (common on this dataset): say so confidently. The relationships here are mostly simple and additive, and a simpler model that's as good is better, because it's easier to explain and maintain.

## 5. Threshold by profit (`src/business.py`)

- A model gives a score from 0 to 1. Someone has to decide which score is "high enough" to act on. 0.5 is a default, not a business decision.
- For each threshold: **value saved** = churners reached × save rate × monthly charge × months retained; **cost** = offers sent × offer cost. Pick the threshold with the highest net.
- Chosen on **out-of-fold predictions** (`cross_val_predict`), so the test set stays clean.
- **Two benchmarks** on the test set: offering to everyone, and offering to the same number of random customers. The second shows exactly what the model adds.
- **Interview line:** "I picked the threshold based on what a missed churner costs vs. a wasted offer, not on accuracy."

## 6. Explainability

- **Permutation importance:** shuffle one feature and measure how much PR-AUC drops. It works for any model and is measured on the test set.
- **Odds ratios** from logistic regression: an odds ratio of 2.0 means that attribute is associated with twice the odds of churning, other things equal.
- **Correlation, not causation:** "month-to-month customers churn more" doesn't mean forcing contracts would stop churn. Testing a contract-upgrade offer would answer that.

## 7. Report generation (`src/report.py`)

- `REPORT.md` is written from the run's numbers, so it can never fall out of date with the model. Change an assumption, rerun, and the report updates.

## Questions an interviewer might ask

- *Why not accuracy?* With 26% churn, "nobody churns" scores 74% accuracy and is useless.
- *How did you avoid overfitting and leakage?* Pipelines inside cross-validation, a test set touched once, and the threshold tuned on out-of-fold predictions.
- *Your probabilities with class weighting aren't calibrated. Does that matter?* For ranking and threshold choice, no. For telling the business "this customer has a 63% chance of leaving", I'd add calibration (`CalibratedClassifierCV`).
- *How would you deploy it?* Score customers monthly, send the top of the at-risk list to the retention team, A/B test the offer to measure the real save rate, and retrain when performance drifts.
- *What would you add?* Calibration, SHAP values for per-customer explanations, and survival analysis to predict *when* customers will leave.
