# Dynamic Pricing & Demand Forecasting Engine

A demand forecasting and promotional pricing recommendation system built on the
Rossmann Store Sales dataset. Predicts store-level demand, estimates the sales
lift from running a promotion, and recommends a promo policy that maximizes
simulated revenue against a static-baseline.

## Framing note

Rossmann does not include a continuous `Price` field — the closest lever is
`Promo` (a binary discount/promotion flag). This project treats **promo
decisions as the pricing lever**: "should we run a promo on store X, on day Y"
stands in for "what price should we set." This is a common and legitimate
simplification for demand-response modeling when granular price data isn't
available, and it's called out explicitly here rather than glossed over.

## Project structure

```
dynamic-pricing-engine/
├── data/
│   ├── raw/            # train.csv, store.csv, test.csv from Kaggle
│   └── processed/      # cleaned/feature-engineered parquet files
├── notebooks/          # EDA and experimentation
├── src/
│   ├── data_prep.py        # cleaning + feature engineering
│   ├── forecast_model.py   # demand forecasting (XGBoost + Prophet)
│   ├── elasticity.py       # promo lift / demand-response estimation
│   ├── pricing_engine.py   # recommendation logic + constraints
│   └── evaluate.py         # backtesting + metrics
├── models/              # saved trained models (.pkl / .json)
├── api/
│   └── main.py          # FastAPI serving layer
├── dashboard/            # React frontend
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Get the data

Download from Kaggle: https://www.kaggle.com/c/rossmann-store-sales/data
Place `train.csv`, `store.csv`, `test.csv` into `data/raw/`.

```bash
# Using the Kaggle CLI (requires ~/.kaggle/kaggle.json credentials)
kaggle competitions download -c rossmann-store-sales -p data/raw
cd data/raw && unzip rossmann-store-sales.zip && cd ../..
```

### 2. Run the pipeline

```bash
python src/data_prep.py          # -> data/processed/features.parquet
python src/forecast_model.py     # trains XGBoost + Prophet, saves to models/
python src/elasticity.py         # estimates promo lift, saves to models/
python src/pricing_engine.py     # generates recommendations
python src/evaluate.py           # backtests, prints MAPE/RMSE + simulated revenue lift
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Run the dashboard

```bash
cd dashboard
npm install
npm run dev
```

## Build phases

1. **EDA & feature engineering** (`data_prep.py`) — seasonality, holidays, store
   clusters, missing-value handling
2. **Demand forecasting** (`forecast_model.py`) — XGBoost with lag/rolling
   features vs. Prophet, compared on MAPE/RMSE
3. **Elasticity / promo response** (`elasticity.py`) — estimated sales lift
   from `Promo=1` vs `Promo=0`, controlling for seasonality and store type
4. **Pricing/promo recommendation** (`pricing_engine.py`) — given forecast +
   lift estimate + a margin/frequency constraint, recommends a promo calendar
   per store; simulates revenue vs. a static baseline on holdout data
5. **Serving + dashboard** (`api/`, `dashboard/`) — FastAPI endpoints, React UI
   with store selector, forecast chart, and recommended action

## Headline metrics to report

Once you've run `evaluate.py`, pull these into your resume/writeup:

- Forecast accuracy: MAPE / RMSE (model vs. seasonal-naive baseline)
- Simulated revenue lift: recommended promo policy vs. static baseline, on
  held-out test period
