"""
Phase 1 — Data cleaning & feature engineering for the Rossmann dataset.

Reads data/raw/{train,store}.csv, joins them, engineers time/seasonality
and store-context features, and writes a clean feature set to
data/processed/features.parquet.

Run: python src/data_prep.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    train = pd.read_csv(RAW_DIR / "train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(RAW_DIR / "store.csv")
    return train, store


def clean_store(store: pd.DataFrame) -> pd.DataFrame:
    store = store.copy()
    # Missing competition distance -> assume far away (no nearby competitor)
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(
        store["CompetitionDistance"].median()
    )
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                "Promo2SinceWeek", "Promo2SinceYear"]:
        store[col] = store[col].fillna(0)
    store["PromoInterval"] = store["PromoInterval"].fillna("")
    return store


def engineer_features(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    df = train.merge(store, on="Store", how="left")

    # Drop closed-store / zero-sales rows — not meaningful for demand modeling
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0=Mon
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Encode categoricals
    df["StateHoliday"] = df["StateHoliday"].replace(0, "0").astype(str)
    for col in ["StoreType", "Assortment", "StateHoliday"]:
        df[col] = df[col].astype("category").cat.codes

    # Competition "months open" feature
    comp_open = pd.to_datetime(
        dict(year=df["CompetitionOpenSinceYear"].replace(0, np.nan),
             month=df["CompetitionOpenSinceMonth"].replace(0, np.nan), day=1),
        errors="coerce",
    )
    df["CompetitionOpenMonths"] = (
        (df["Date"] - comp_open).dt.days / 30
    ).clip(lower=0).fillna(0)

    # Sort for lag/rolling features
    df = df.sort_values(["Store", "Date"])

    # Lag + rolling demand features per store
    df["Sales_lag_7"] = df.groupby("Store")["Sales"].shift(7)
    df["Sales_lag_14"] = df.groupby("Store")["Sales"].shift(14)
    df["Sales_roll_mean_7"] = (
        df.groupby("Store")["Sales"].shift(1).rolling(7).mean().reset_index(0, drop=True)
    )
    df["Sales_roll_mean_28"] = (
        df.groupby("Store")["Sales"].shift(1).rolling(28).mean().reset_index(0, drop=True)
    )

    # Drop rows where lag features are NaN (start of each store's series)
    df = df.dropna(subset=["Sales_lag_7", "Sales_lag_14", "Sales_roll_mean_7", "Sales_roll_mean_28"])

    feature_cols = [
        "Store", "DayOfWeek", "IsWeekend", "Promo", "StateHoliday", "SchoolHoliday",
        "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenMonths",
        "Promo2", "Year", "Month", "Day", "WeekOfYear",
        "Sales_lag_7", "Sales_lag_14", "Sales_roll_mean_7", "Sales_roll_mean_28",
        "Customers", "Sales", "Date",
    ]
    return df[feature_cols].reset_index(drop=True)


def main():
    print("Loading raw data...")
    train, store = load_raw()
    store = clean_store(store)

    print("Engineering features...")
    features = engineer_features(train, store)

    out_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Wrote {len(features):,} rows to {out_path}")
    print(features.head())


if __name__ == "__main__":
    main()
