"""
Turn model scores into money.

The question a retention team actually asks is not "how accurate is the model?" but
"if we offer a retention deal to everyone above this risk score, do we make or lose money?"

Assumptions (all editable in config.py, and stated in the report):
  - Every targeted customer gets an offer that costs OFFER_COST.
  - A customer who was going to churn accepts and stays with probability SAVE_RATE.
  - A saved customer is worth their monthly charge for MONTHS_RETAINED more months.
  - Offers to customers who weren't going to leave are pure cost.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Assumptions:
    offer_cost: float = 50.0        # e.g. a one-off discount or bill credit
    save_rate: float = 0.30         # share of would-be churners the offer keeps
    months_retained: int = 12       # value horizon for a saved customer


def campaign_profit(y_true, scores, monthly_charges, threshold: float, a: Assumptions) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    monthly = np.asarray(monthly_charges, dtype=float)

    targeted = scores >= threshold
    true_pos = targeted & (y_true == 1)
    value_saved = (a.save_rate * monthly[true_pos] * a.months_retained).sum()
    cost = a.offer_cost * targeted.sum()
    return {
        "threshold": round(float(threshold), 3),
        "customers_targeted": int(targeted.sum()),
        "churners_reached": int(true_pos.sum()),
        "churners_total": int((y_true == 1).sum()),
        "value_saved": round(float(value_saved), 2),
        "campaign_cost": round(float(cost), 2),
        "net_profit": round(float(value_saved - cost), 2),
    }


def profit_curve(y_true, scores, monthly_charges, a: Assumptions, thresholds=None) -> list[dict]:
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2) if thresholds is None else thresholds
    return [campaign_profit(y_true, scores, monthly_charges, t, a) for t in thresholds]


def best_threshold(y_true, scores, monthly_charges, a: Assumptions) -> dict:
    return max(profit_curve(y_true, scores, monthly_charges, a), key=lambda r: r["net_profit"])
