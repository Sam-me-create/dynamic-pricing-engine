import React, { useEffect, useState, useMemo } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
} from "recharts";
import "./App.css";

const API_BASE = "/api";

function useFetch(path, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}${path}`)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading };
}

function fmtNum(n) {
  if (n === null || n === undefined) return "—";
  return Math.round(n).toLocaleString();
}

function fmtPct(n) {
  if (n === null || n === undefined) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
}

function Header() {
  return (
    <header className="header">
      <div className="header-mark">PDE</div>
      <div className="header-text">
        <h1>Pricing &amp; Demand Console</h1>
        <p>Store-level demand forecasts and promo recommendations, Rossmann holdout period</p>
      </div>
    </header>
  );
}

function SummaryStrip() {
  const { data, error, loading } = useFetch("/summary");

  if (loading) return <div className="summary-strip loading">Loading model summary…</div>;
  if (error)
    return (
      <div className="summary-strip error">
        Model outputs not available yet ({error}). Run the pipeline in src/ first.
      </div>
    );

  const xgb = data.forecast_metrics.find((m) => m.label === "XGBoost");
  const baseline = data.forecast_metrics.find((m) => m.label.includes("baseline"));
  const sim = data.revenue_simulation;

  return (
    <div className="summary-strip">
      <div className="summary-cell">
        <span className="summary-label">Forecast error (MAPE)</span>
        <span className="summary-value">{xgb?.mape.toFixed(1)}%</span>
        <span className="summary-sub">
          vs {baseline?.mape.toFixed(1)}% seasonal-naive baseline
        </span>
      </div>
      <div className="summary-divider" />
      <div className="summary-cell">
        <span className="summary-label">Projected revenue lift</span>
        <span className="summary-value accent">{fmtPct(sim?.lift_vs_no_promo_pct)}</span>
        <span className="summary-sub">recommended promo policy vs. static no-promo</span>
      </div>
      <div className="summary-divider" />
      <div className="summary-cell">
        <span className="summary-label">Holdout revenue simulated</span>
        <span className="summary-value">{fmtNum(sim?.recommended_policy_revenue)}</span>
        <span className="summary-sub">under the constrained promo policy</span>
      </div>
    </div>
  );
}

function StoreSelector({ stores, selected, onSelect }) {
  return (
    <div className="store-selector">
      <label htmlFor="store-select">Store</label>
      <select
        id="store-select"
        value={selected ?? ""}
        onChange={(e) => onSelect(Number(e.target.value))}
      >
        {stores.map((s) => (
          <option key={s} value={s}>
            Store {s}
          </option>
        ))}
      </select>
    </div>
  );
}

function ForecastChart({ storeId }) {
  const { data, error, loading } = useFetch(
    storeId ? `/forecast/${storeId}` : null,
    [storeId]
  );
  const { data: recData } = useFetch(
    storeId ? `/recommend/${storeId}` : null,
    [storeId]
  );

  const merged = useMemo(() => {
    if (!data) return [];
    const recByDate = new Map((recData ?? []).map((r) => [r.date, r]));
    return data.map((d) => ({
      ...d,
      recommended_promo: recByDate.get(d.date)?.recommended_promo ?? 0,
    }));
  }, [data, recData]);

  if (!storeId) return null;
  if (loading) return <div className="panel loading">Loading forecast…</div>;
  if (error)
    return (
      <div className="panel error">
        No forecast data for this store yet ({error}).
      </div>
    );

  return (
    <div className="panel">
      <div className="panel-heading">
        <h2>Actual vs. forecast demand</h2>
        <div className="legend">
          <span>
            <i className="dot dot-actual" /> Actual sales
          </span>
          <span>
            <i className="dot dot-forecast" /> Forecast (XGBoost)
          </span>
          <span>
            <i className="dot dot-promo" /> Recommended promo day
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={merged} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#dcd6c9" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fontFamily: "IBM Plex Mono", fontSize: 11, fill: "#4a5568" }}
            tickLine={false}
            axisLine={{ stroke: "#dcd6c9" }}
            minTickGap={30}
          />
          <YAxis
            tick={{ fontFamily: "IBM Plex Mono", fontSize: 11, fill: "#4a5568" }}
            tickLine={false}
            axisLine={false}
            width={56}
          />
          <Tooltip
            contentStyle={{
              fontFamily: "IBM Plex Sans",
              fontSize: 13,
              border: "1px solid #dcd6c9",
              borderRadius: 4,
              background: "#fcfbf8",
            }}
            labelStyle={{ fontFamily: "IBM Plex Mono", fontSize: 11 }}
          />
          {merged.map((d, i) =>
            d.recommended_promo ? (
              <ReferenceArea
                key={i}
                x1={d.date}
                x2={d.date}
                fill="#b8763e"
                fillOpacity={0.12}
                stroke="none"
              />
            ) : null
          )}
          <Line
            type="monotone"
            dataKey="actual_sales"
            stroke="#14181f"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="predicted_sales"
            stroke="#b8763e"
            strokeWidth={2}
            strokeDasharray="4 3"
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function RecommendationTable({ storeId }) {
  const { data, error, loading } = useFetch(
    storeId ? `/recommend/${storeId}` : null,
    [storeId]
  );

  if (!storeId) return null;
  if (loading) return <div className="panel loading">Loading recommendations…</div>;
  if (error)
    return (
      <div className="panel error">
        No recommendation data for this store yet ({error}).
      </div>
    );

  const promoDays = data.filter((d) => d.recommended_promo === 1);

  return (
    <div className="panel">
      <div className="panel-heading">
        <h2>Recommended promo days</h2>
        <span className="panel-sub">{promoDays.length} of {data.length} holdout days</span>
      </div>
      {promoDays.length === 0 ? (
        <p className="empty-state">
          No promo days recommended for this store in the holdout window.
        </p>
      ) : (
        <table className="rec-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Expected incremental sales</th>
            </tr>
          </thead>
          <tbody>
            {promoDays.map((d) => (
              <tr key={d.date}>
                <td className="mono">{d.date}</td>
                <td className="mono accent">
                  +{fmtNum(d.expected_incremental_sales)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function App() {
  const { data: stores, error: storesError, loading: storesLoading } = useFetch("/stores");
  const [selectedStore, setSelectedStore] = useState(null);

  useEffect(() => {
    if (stores && stores.length > 0 && selectedStore === null) {
      setSelectedStore(stores[0]);
    }
  }, [stores, selectedStore]);

  return (
    <div className="app">
      <Header />
      <SummaryStrip />

      <main className="main">
        {storesLoading && <div className="panel loading">Loading store list…</div>}
        {storesError && (
          <div className="panel error">
            Can't reach the API yet ({storesError}). Start it with:
            <code> uvicorn api.main:app --reload --port 8000</code>
          </div>
        )}
        {stores && stores.length > 0 && (
          <>
            <StoreSelector
              stores={stores}
              selected={selectedStore}
              onSelect={setSelectedStore}
            />
            <ForecastChart storeId={selectedStore} />
            <RecommendationTable storeId={selectedStore} />
          </>
        )}
      </main>

      <footer className="footer">
        Promo lift is modeled from historical Promo=1 vs. Promo=0 sales, not
        measured via a live A/B test — treat projected figures as simulated.
      </footer>
    </div>
  );
}
