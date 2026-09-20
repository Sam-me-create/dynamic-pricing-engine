"""
Phase 5 — Consolidated evaluation report.

Pulls together forecast_metrics.csv and revenue_simulation.csv into one
printed summary — this is what you pull numbers from for your resume bullet
and interview talking points.

Run: python src/evaluate.py
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"


def main():
    print("=" * 60)
    print("FORECAST ACCURACY")
    print("=" * 60)
    metrics = pd.read_csv(MODELS / "forecast_metrics.csv")
    print(metrics.to_string(index=False))

    xgb_row = metrics[metrics["label"] == "XGBoost"].iloc[0]
    baseline_row = metrics[metrics["label"] == "Seasonal-naive baseline"].iloc[0]
    mape_improvement = baseline_row["mape"] - xgb_row["mape"]
    print(f"\nXGBoost improves MAPE by {mape_improvement:.2f} percentage points "
          f"vs. seasonal-naive baseline.")

    print("\n" + "=" * 60)
    print("SIMULATED REVENUE IMPACT (holdout period)")
    print("=" * 60)
    sim = pd.read_csv(MODELS / "revenue_simulation.csv").iloc[0]
    print(f"Actual revenue (holdout):          {sim['actual_revenue_holdout']:,.0f}")
    print(f"Static no-promo baseline:          {sim['static_no_promo_revenue']:,.0f}")
    print(f"Recommended promo policy:          {sim['recommended_policy_revenue']:,.0f}")
    print(f"  -> Lift vs. no-promo baseline:   {sim['lift_vs_no_promo_pct']:.2f}%")
    print(f"Always-promo (unconstrained) rev:  {sim['static_always_promo_revenue']:,.0f}")
    print(f"  -> Gap vs. always-promo:         {sim['gap_vs_always_promo_pct']:.2f}% "
          f"(this is the value of the margin constraint)")

    print("\n" + "=" * 60)
    print("RESUME/WRITEUP TALKING POINTS")
    print("=" * 60)
    print(f"- Built a demand forecasting model (XGBoost) achieving {xgb_row['mape']:.1f}% MAPE,")
    print(f"  a {mape_improvement:.1f}-point improvement over a seasonal-naive baseline.")
    print(f"- Estimated promo-driven demand lift and built a constrained recommendation")
    print(f"  engine projecting a {sim['lift_vs_no_promo_pct']:.1f}% revenue lift vs. a")
    print(f"  static no-promo policy, under a monthly promo-day margin constraint.")
    print(f"- IMPORTANT: label this as a *simulated/projected* lift from a forecasting +")
    print(f"  lift-estimation model, not a measured A/B test result, when you talk about it.")


if __name__ == "__main__":
    main()
