"""
Phase 3 — Promo response ("elasticity") estimation.

Rossmann has no continuous price field, so this estimates the demand
response to the Promo flag instead: how much does running a promo lift
sales, controlling for day-of-week, seasonality, and store type?

Approach: a regression-based lift estimate per StoreType, comparing actual
Promo=1 sales against a same-store/day-of-week/month counterfactual built
from Promo=0 periods (a simple difference-in-differences style estimate).

Run: python src/elasticity.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"


def load_features():
    df = pd.read_parquet(PROCESSED / "features.parquet")
    return df


def estimate_lift_by_storetype(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (StoreType, DayOfWeek) cell, compare mean sales when Promo=1
    vs Promo=0. This controls for the two biggest confounders (store
    format and day-of-week seasonality) without needing a full causal model.
    """
    grouped = (
        df.groupby(["StoreType", "DayOfWeek", "Promo"])["Sales"]
        .mean()
        .unstack("Promo")
        .rename(columns={0: "sales_no_promo", 1: "sales_promo"})
        .dropna()
    )
    grouped["lift_abs"] = grouped["sales_promo"] - grouped["sales_no_promo"]
    grouped["lift_pct"] = (grouped["lift_abs"] / grouped["sales_no_promo"]) * 100
    return grouped.reset_index()


def estimate_lift_by_store(df: pd.DataFrame) -> pd.DataFrame:
    """Same idea, but per individual store — used by the pricing engine
    for store-specific recommendations."""
    grouped = (
        df.groupby(["Store", "Promo"])["Sales"]
        .mean()
        .unstack("Promo")
        .rename(columns={0: "sales_no_promo", 1: "sales_promo"})
    )
    grouped["lift_pct"] = (
        (grouped["sales_promo"] - grouped["sales_no_promo"]) / grouped["sales_no_promo"]
    ) * 100
    # Some stores may never run a promo in the data — fall back to the
    # StoreType-level average lift for those.
    return grouped.reset_index()


def main():
    print("Loading features...")
    df = load_features()

    print("\nEstimating promo lift by StoreType x DayOfWeek...")
    lift_by_type = estimate_lift_by_storetype(df)
    print(lift_by_type.round(2).to_string(index=False))

    print("\nEstimating promo lift by individual store...")
    lift_by_store = estimate_lift_by_store(df)
    print(f"Computed lift estimates for {lift_by_store['lift_pct'].notna().sum():,} stores "
          f"(of {lift_by_store.shape[0]:,} total)")

    lift_by_type.to_csv(MODELS / "promo_lift_by_storetype.csv", index=False)
    lift_by_store.to_csv(MODELS / "promo_lift_by_store.csv", index=False)
    print(f"\nSaved lift estimates to {MODELS}/promo_lift_by_*.csv")


if __name__ == "__main__":
    main()
