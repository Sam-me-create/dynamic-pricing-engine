"""
FastAPI serving layer for the dynamic pricing engine.

Endpoints:
  GET /stores                       -> list of store IDs available in holdout data
  GET /forecast/{store_id}          -> forecast vs actual sales for the holdout period
  GET /recommend/{store_id}         -> recommended promo calendar for the store
  GET /summary                      -> overall model + revenue simulation metrics

Run: uvicorn api.main:app --reload --port 8000
"""

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"

app = FastAPI(title="Dynamic Pricing Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{path.name} not found — run the src/ pipeline scripts first "
                    "(data_prep.py -> forecast_model.py -> elasticity.py -> pricing_engine.py).",
        )
    return pd.read_parquet(path)


class ForecastPoint(BaseModel):
    date: str
    actual_sales: float
    predicted_sales: float
    promo_actual: int


class RecommendationPoint(BaseModel):
    date: str
    recommended_promo: int
    expected_incremental_sales: float


@app.get("/stores", response_model=List[int])
def list_stores():
    df = _load_parquet_safe(PROCESSED / "holdout_predictions.parquet")
    return sorted(df["Store"].unique().tolist())


@app.get("/forecast/{store_id}", response_model=List[ForecastPoint])
def get_forecast(store_id: int):
    df = _load_parquet_safe(PROCESSED / "holdout_predictions.parquet")
    store_df = df[df["Store"] == store_id].sort_values("Date")
    if store_df.empty:
        raise HTTPException(status_code=404, detail=f"No holdout data for store {store_id}")
    return [
        ForecastPoint(
            date=str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"]),
            actual_sales=float(row["Sales"]),
            predicted_sales=float(row["pred_xgboost"]),
            promo_actual=int(row["Promo"]),
        )
        for _, row in store_df.iterrows()
    ]


@app.get("/recommend/{store_id}", response_model=List[RecommendationPoint])
def get_recommendation(store_id: int):
    df = _load_parquet_safe(PROCESSED / "promo_recommendations.parquet")
    store_df = df[df["Store"] == store_id].sort_values("Date")
    if store_df.empty:
        raise HTTPException(status_code=404, detail=f"No recommendation data for store {store_id}")
    return [
        RecommendationPoint(
            date=str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"]),
            recommended_promo=int(row["recommended_promo"]),
            expected_incremental_sales=float(row["expected_incremental"]),
        )
        for _, row in store_df.iterrows()
    ]


@app.get("/summary")
def get_summary():
    metrics_path = MODELS / "forecast_metrics.csv"
    sim_path = MODELS / "revenue_simulation.csv"
    if not metrics_path.exists() or not sim_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Metrics not found — run src/forecast_model.py and src/pricing_engine.py first.",
        )
    metrics = pd.read_csv(metrics_path).to_dict(orient="records")
    sim = pd.read_csv(sim_path).iloc[0].to_dict()
    return {"forecast_metrics": metrics, "revenue_simulation": sim}


@app.get("/")
def root():
    return {"status": "ok", "message": "Dynamic Pricing Engine API. See /docs for endpoints."}
