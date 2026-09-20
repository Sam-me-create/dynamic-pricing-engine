# Notebooks

Use this folder for:
- `01_eda.ipynb` — exploratory analysis on the raw Rossmann data (seasonality,
  promo frequency, store-type distributions)
- `02_prophet_comparison.ipynb` — Prophet forecasts for a handful of sample
  stores, compared against the XGBoost model from `src/forecast_model.py`
  (Prophet is too slow to loop over all 1,115 stores in a script, so this
  comparison is best done interactively on a sample)
