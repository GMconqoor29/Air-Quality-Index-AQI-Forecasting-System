/**
 * AQI API — Mock implementation
 *
 * All functions return promises that resolve with simulated data, but they are
 * generated from a single memoized time series so the dashboard reads as one
 * coherent snapshot (the current reading is the tail of the historical series,
 * and the forecast extends it).
 *
 * When the Flask backend is ready, replace the function bodies with real axios
 * calls — the signatures stay the same.
 *
 * Base URL proxied through Vite: /api → http://localhost:5000
 */

import axios from 'axios';

// Axios instance (ready for when the backend exists)
const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
});

/* ─────────────────────────────────────────────
   Helpers
───────────────────────────────────────────── */

/** Resolve after `ms` milliseconds (simulates network latency). */
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Clamp a value between lo and hi. */
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Round to n decimal places. */
const round = (v, n = 1) => Math.round(v * 10 ** n) / 10 ** n;

/**
 * Derive approximate PM2.5 from AQI (simplified linear for AQI 51–150 range).
 * For display purposes only.
 */
function aqiToPM25(aqi) {
  if (aqi <= 50) return round(6 + (aqi / 50) * 6.1, 1);
  if (aqi <= 100) return round(12.1 + ((aqi - 51) / 49) * (35.4 - 12.1), 1);
  if (aqi <= 150) return round(35.5 + ((aqi - 101) / 49) * (55.4 - 35.5), 1);
  if (aqi <= 200) return round(55.5 + ((aqi - 151) / 49) * (150.4 - 55.5), 1);
  return round(150.5 + ((aqi - 201) / 99) * (250.4 - 150.5), 1);
}

/* ─────────────────────────────────────────────
   Coherent mock time series (built once per session)
───────────────────────────────────────────── */

const SERIES_HOURS = 168; // 7 days, hourly

let _series = null;

/**
 * Build (and memoize) a realistic 7-day hourly Hanoi series. Anchored to the
 * current hour so the tail of the series is "now".
 */
function getSeries() {
  if (_series) return _series;

  const now = new Date();
  now.setMinutes(0, 0, 0);
  const oneHour = 3600 * 1000;

  // Small deterministic PRNG keeps the shape stable across re-renders.
  let seed = 20261027;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  const points = [];
  let aqi = 118;

  for (let i = SERIES_HOURS - 1; i >= 0; i--) {
    const ts = new Date(now.getTime() - i * oneHour);
    const h = ts.getHours();

    // Diurnal bias: rush hours + evening are higher, overnight is lower.
    let bias = 0;
    if ((h >= 7 && h <= 9) || (h >= 18 && h <= 21)) bias = 22;
    else if (h >= 0 && h <= 5) bias = -18;
    else if (h >= 10 && h <= 17) bias = 4;
    else bias = -4;

    const walk = (rnd() - 0.5) * 18;
    aqi = clamp(aqi + walk + bias * 0.25 + (rnd() - 0.5) * 6, 35, 260);
    if (rnd() < 0.02) aqi = clamp(aqi + 60, 120, 320);

    const aqiRounded = Math.round(aqi);
    const pm25 = aqiToPM25(aqiRounded);
    const prevAqi = points.length > 0 ? points[points.length - 1].aqi : aqiRounded;
    const predictedAqi = Math.round(clamp(prevAqi * 0.72 + aqiRounded * 0.28 + (rnd() - 0.5) * 10, 25, 330));

    points.push({
      timestamp: ts.toISOString(),
      aqi: aqiRounded,
      predictedAqi,
      pm25,
      pm10: round(pm25 * (1.5 + rnd() * 0.4), 1),
      temperature: round(25 + 5 * Math.sin(((h - 9) / 24) * 2 * Math.PI) + (rnd() - 0.5) * 2, 1),
      relativeHumidity: round(70 + 14 * Math.sin(((h - 6) / 24) * 2 * Math.PI) + (rnd() - 0.5) * 6, 1),
      pressure: round(1012 + (rnd() - 0.5) * 6, 1),
    });
  }

  _series = { points, now };
  return _series;
}

/* ─────────────────────────────────────────────
   Mock model comparison data (from the paper)
───────────────────────────────────────────── */
const MODEL_COMPARISON_DATA = [
  { model: 'Persistence',          rmse: 31.24, mae: 18.91, peakRecall: 0.692, peakMAE: 26.96, amse: 2533.0 },
  { model: 'Base LSTM',            rmse: 30.47, mae: 22.04, peakRecall: 0.393, peakMAE: 39.45, amse: 2721.7 },
  { model: 'GRU',                  rmse: 30.24, mae: 21.95, peakRecall: 0.375, peakMAE: 39.75, amse: 2672.6 },
  { model: 'Stacked LSTM',         rmse: 29.44, mae: 20.69, peakRecall: 0.722, peakMAE: 27.72, amse: 2003.8 },
  { model: 'Tuned XGBoost',        rmse: 28.66, mae: 19.82, peakRecall: 0.621, peakMAE: 31.33, amse: 2281.9 },
  { model: 'XGBoost',              rmse: 28.42, mae: 19.66, peakRecall: 0.639, peakMAE: 31.14, amse: 2257.4 },
  { model: 'Soft-Gated Ensemble',  rmse: 31.67, mae: 22.67, peakRecall: 0.834, peakMAE: 20.57, amse: 1664.3 },
  { model: 'CNN-LSTM Attention',   rmse: 32.53, mae: 23.25, peakRecall: 0.850, peakMAE: 20.41, amse: 1714.8 },
  { model: 'CNN-GRU Attention',    rmse: 33.43, mae: 23.46, peakRecall: 0.857, peakMAE: 21.01, amse: 1776.7 },
  { model: 'Two-Stage Regressor',  rmse: 33.90, mae: 23.10, peakRecall: 0.904, peakMAE: 21.68, amse: 1979.2 },
];

/** Models that can emit a point forecast (used by the forecast selector). */
export const FORECAST_MODELS = [
  'Best available',
  'Tuned XGBoost',
  'XGBoost',
  'Soft-Gated Ensemble',
  'Two-Stage Regressor',
  'CNN-LSTM Attention',
  'CNN-GRU Attention',
  'GRU',
  'Stacked LSTM',
  'Base LSTM',
];

/* ─────────────────────────────────────────────
   Mock alert history
───────────────────────────────────────────── */
function generateAlertHistory() {
  const { points } = getSeries();
  const alerts = [];
  const oneHour = 3600 * 1000;

  // Anchor alerts to real spikes in the series when possible.
  const spikes = points
    .map((p, i) => ({ aqi: p.aqi, i }))
    .filter((s) => s.aqi >= 180)
    .slice(-6)
    .reverse();

  const templates = [
    { type: 'hazardous', message: 'AQI reached Hazardous level' },
    { type: 'very_unhealthy', message: 'AQI reached Very Unhealthy level' },
    { type: 'forecast_high', message: 'Forecast AQI ≥ 200 — Very Unhealthy expected' },
    { type: 'very_unhealthy', message: 'AQI reached Very Unhealthy level' },
    { type: 'forecast_high', message: 'Forecast AQI ≥ 200 — Very Unhealthy expected' },
    { type: 'forecast_high', message: 'Forecast AQI ≥ 200 — Very Unhealthy expected' },
  ];

  templates.forEach((t, idx) => {
    const anchor = spikes[idx] ?? { aqi: 200, i: 6 };
    alerts.push({
      id: `alert-${idx}`,
      ...t,
      aqiValue: Math.max(anchor.aqi, 180),
      timestamp: new Date(points[anchor.i].timestamp).toISOString(),
    });
  });

  return alerts;
}

/* ─────────────────────────────────────────────
   Public API functions (Connected to Real-Time Backend /api -> :5000)
───────────────────────────────────────────── */

/**
 * Trigger an immediate OpenAQ + Sensor synchronization on the backend.
 */
export async function syncTelemetryBackend() {
  try {
    const res = await api.post('/sync');
    return res.data;
  } catch {
    return null;
  }
}

/**
 * Fetch the current sensor reading (from ESP32 / OpenAQ Location 4946813).
 */
export async function fetchCurrentReading() {
  try {
    const res = await api.get('/current');
    if (res.data && typeof res.data.aqi === 'number') {
      return res.data;
    }
  } catch {
    // Fallback to simulated series if backend is offline
  }
  await delay(120);
  const { points } = getSeries();
  return { ...points[points.length - 1] };
}

/**
 * Fetch the AQI forecast at the given horizon from the chosen trained model.
 *
 * @param {object} [opts]
 * @param {1|6} [opts.horizon=1] Forecast horizon in hours.
 * @param {string} [opts.model='Best available'] Model name.
 */
export async function fetchForecast({ horizon = 1, model = 'Best available' } = {}) {
  try {
    const res = await api.get('/forecast', { params: { horizon, model } });
    if (res.data && typeof res.data.predicted_aqi === 'number') {
      return res.data;
    }
  } catch {
    // Fallback to simulated forecast if backend is offline
  }
  await delay(180);
  const { points } = getSeries();
  const last = points[points.length - 1];

  const drift = horizon === 6 ? round((last.aqi - points[points.length - 7].aqi) * 0.6, 0) : 0;
  const predicted_aqi = Math.round(clamp(last.aqi + drift, 20, 350));
  const margin = horizon === 6 ? round(16 + Math.abs(drift) * 0.4, 0) : round(10, 0);

  const model_used = model === 'Best available' ? 'Tuned XGBoost' : model;

  return {
    predicted_aqi,
    model_used,
    confidence_interval: [
      Math.max(0, predicted_aqi - margin),
      Math.min(500, predicted_aqi + margin),
    ],
    horizon,
  };
}

/**
 * Fetch historical sensor readings with rolling t+1 predictions for the chosen model.
 *
 * @param {number} hours Number of past hours to retrieve (default 48).
 * @param {string} [model='Best available'] Model name for rolling t+1 predictions.
 */
export async function fetchHistoricalData(hours = 48, model = 'Best available') {
  try {
    const res = await api.get('/historical', { params: { hours, model } });
    if (Array.isArray(res.data) && res.data.length > 0) {
      return res.data;
    }
  } catch {
    // Fallback to simulated series if backend is offline
  }
  await delay(200);
  const { points } = getSeries();
  return points.slice(-hours);
}

/**
 * Fetch model comparison metrics from the backend.
 */
export async function fetchModelComparison() {
  try {
    const res = await api.get('/model-comparison');
    if (Array.isArray(res.data) && res.data.length > 0) {
      return res.data;
    }
  } catch {
    // Fallback
  }
  await delay(140);
  return MODEL_COMPARISON_DATA;
}

/**
 * Fetch alert history (last N alerts).
 */
export async function fetchAlertHistory() {
  try {
    const res = await api.get('/alerts');
    if (Array.isArray(res.data)) {
      return res.data;
    }
  } catch {
    // Fallback
  }
  await delay(140);
  return generateAlertHistory();
}

export default api;
