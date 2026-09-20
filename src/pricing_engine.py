"""
Phase 4 — Promo recommendation engine.

Combines the demand forecast (forecast_model.py) with the promo lift
estimates (elasticity.py) to recommend which days each store should run a
promo, subject to a simple business constraint (max promo days per store
per month, to model a real "you can't discount every day" margin
constraint). Outputs a recommended promo calendar and a naive projected
revenue comparison vs. a static (never-promo / always-promo) baseline.

Run: python src/pricing_engine.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"

MAX_PROMO_DAYS_PER_MONTH = 10  # business constraint: margin floor


def load_inputs():
    holdout = pd.read_parquet(PROCESSED / "holdout_predictions.parquet")
    lift_by_store = pd.read_csv(MODELS / "promo_lift_by_store.csv")
    lift_by_type = pd.read_csv(MODELS / "promo_lift_by_storetype.csv")
    return holdout, lift_by_store, lift_by_type


def get_store_lift(store_id, storetype, dow, lift_by_store, lift_by_type):
    row = lift_by_store[lift_by_store["Store"] == store_id]
    if not row.empty and pd.notna(row.iloc[0]["lift_pct"]):
        return row.iloc[0]["lift_pct"]
    # fallback to StoreType/DayOfWeek average
    fallback = lift_by_type[
        (lift_by_type["StoreType"] == storetype) & (lift_by_type["DayOfWeek"] == dow)
    ]
    return fallback.iloc[0]["lift_pct"] if not fallback.empty else 0.0


def recommend_promo_calendar(holdout: pd.DataFrame, lift_by_store, lift_by_type) -> pd.DataFrame:
    df = holdout.copy()
    df["storetype"] = df["StoreType"]

    df["est_lift_pct"] = df.apply(
        lambda r: get_store_lift(r["Store"], r["storetype"], r["DayOfWeek"],
                                  lift_by_store, lift_by_type),
        axis=1,
    )
    # Expected incremental sales if a promo were run this day, using the
    # forecast (pred_xgboost) as the "no promo" baseline demand estimate.
    df["expected_promo_sales"] = df["pred_xgboost"] * (1 + df["est_lift_pct"] / 100)
    df["expected_incremental"] = df["expected_promo_sales"] - df["pred_xgboost"]

    # Rank days within each store-month by incremental value, cap at the
    # monthly promo-day constraint.
    df["year_month"] = df["Date"].dt.to_period("M").astype(str)
    df["rank_in_month"] = (
        df.groupby(["Store", "year_month"])["expected_incremental"]
        .rank(method="first", ascending=False)
    )
    df["recommended_promo"] = (df["rank_in_month"] <= MAX_PROMO_DAYS_PER_MONTH).astype(int)
    return df


def simulate_revenue(df: pd.DataFrame) -> dict:
    """Compare simulated revenue under three policies against ACTUAL sales:
       - static_no_promo: never run a promo
       - static_always_promo: promo every day (unrealistic, upper bound / margin-ignoring)
       - recommended: this engine's constrained recommendation
    """
    no_promo_rev = df["pred_xgboost"].sum()
    always_promo_rev = df["expected_promo_sales"].sum()
    recommended_rev = np.where(
        df["recommended_promo"] == 1, df["expected_promo_sales"], df["pred_xgboost"]
    ).sum()
    actual_rev = df["Sales"].sum()

    return {
        "actual_revenue_holdout": actual_rev,
        "static_no_promo_revenue": no_promo_rev,
        "static_always_promo_revenue": always_promo_rev,
        "recommended_policy_revenue": recommended_rev,
        "lift_vs_no_promo_pct": (recommended_rev - no_promo_rev) / no_promo_rev * 100,
        "gap_vs_always_promo_pct": (always_promo_rev - recommended_rev) / recommended_rev * 100,
    }


def main():
    print("Loading forecast + elasticity outputs...")
    holdout, lift_by_store, lift_by_type = load_inputs()
    holdout["Date"] = pd.to_datetime(holdout["Date"])

    print(f"Generating promo calendar (max {MAX_PROMO_DAYS_PER_MONTH} promo days/store/month)...")
    recs = recommend_promo_calendar(holdout, lift_by_store, lift_by_type)

    out_path = PROCESSED / "promo_recommendations.parquet"
    recs.to_parquet(out_path, index=False)
    print(f"Saved recommendations to {out_path}")

    print("\nSimulating revenue impact on holdout period...")
    sim = simulate_revenue(recs)
    for k, v in sim.items():
        print(f"  {k:32s}: {v:,.1f}" if "pct" not in k else f"  {k:32s}: {v:,.2f}%")

    pd.DataFrame([sim]).to_csv(MODELS / "revenue_simulation.csv", index=False)
    print(f"\nSaved summary to {MODELS / 'revenue_simulation.csv'}")
    print("\nNote: this is a simulation using forecast + lift estimates, not an A/B")
    print("test — call this out explicitly in your writeup as a modeled/projected")
    print("lift, not a measured one.")


if __name__ == "__main__":
    main()
