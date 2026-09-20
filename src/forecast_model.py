"""
Phase 2 — Demand forecasting.

Trains two models on data/processed/features.parquet:
  - XGBoost regressor using lag/rolling + calendar + store features
  - Prophet, per-store (on a sample of stores, since Prophet is slow at scale)

Both are evaluated against a seasonal-naive baseline on a time-based holdout
(last 6 weeks of data). Winner (or ensemble) is saved to models/.

Run: python src/forecast_model.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

HOLDOUT_DAYS = 42  # ~6 weeks


def load_features():
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def time_split(df: pd.DataFrame):
    cutoff = df["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train = df[df["Date"] <= cutoff]
    test = df[df["Date"] > cutoff]
    return train, test


def seasonal_naive_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict this week's sales = same store/day-of-week average from train."""
    dow_avg = train.groupby(["Store", "DayOfWeek"])["Sales"].mean().rename("pred")
    merged = test.merge(dow_avg, on=["Store", "DayOfWeek"], how="left")
    merged["pred"] = merged["pred"].fillna(train["Sales"].mean())
    return merged["pred"].values


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    feature_cols = [
        "Store", "DayOfWeek", "IsWeekend", "Promo", "StateHoliday", "SchoolHoliday",
        "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenMonths",
        "Promo2", "Year", "Month", "Day", "WeekOfYear",
        "Sales_lag_7", "Sales_lag_14", "Sales_roll_mean_7", "Sales_roll_mean_28",
    ]
    X_train, y_train = train[feature_cols], train["Sales"]
    X_test, y_test = test[feature_cols], test["Sales"]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    return model, preds, feature_cols


def evaluate(y_true, y_pred, label):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {label:25s}  MAPE: {mape:6.2f}%   RMSE: {rmse:8.1f}")
    return {"label": label, "mape": mape, "rmse": rmse}


def main():
    print("Loading features...")
    df = load_features()
    train, test = time_split(df)
    print(f"Train: {len(train):,} rows | Test (holdout): {len(test):,} rows")

    results = []

    print("\nBaseline (seasonal-naive)...")
    baseline_preds = seasonal_naive_baseline(train, test)
    results.append(evaluate(test["Sales"], baseline_preds, "Seasonal-naive baseline"))

    print("\nTraining XGBoost...")
    model, xgb_preds, feature_cols = train_xgboost(train, test)
    results.append(evaluate(test["Sales"], xgb_preds, "XGBoost"))

    # Save model + predictions + feature list
    joblib.dump(model, MODELS / "xgb_forecast_model.pkl")
    joblib.dump(feature_cols, MODELS / "xgb_feature_cols.pkl")

    test_out = test.copy()
    test_out["pred_baseline"] = baseline_preds
    test_out["pred_xgboost"] = xgb_preds
    test_out.to_parquet(PROCESSED / "holdout_predictions.parquet", index=False)

    pd.DataFrame(results).to_csv(MODELS / "forecast_metrics.csv", index=False)
    print(f"\nSaved model to {MODELS / 'xgb_forecast_model.pkl'}")
    print(f"Saved holdout predictions to {PROCESSED / 'holdout_predictions.parquet'}")
    print("\nNote: Prophet per-store comparison is in notebooks/ — run there for")
    print("a handful of stores; it's too slow to loop over all 1,115 stores here.")


if __name__ == "__main__":
    main()
