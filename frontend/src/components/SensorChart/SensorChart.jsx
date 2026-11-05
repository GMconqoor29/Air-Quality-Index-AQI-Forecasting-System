import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  ComposedChart,
  CartesianGrid,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { FiArrowDownRight, FiArrowRight, FiArrowUpRight, FiCpu } from 'react-icons/fi';
import { FORECAST_MODELS, fetchForecast, fetchHistoricalData } from '../../api/aqiApi';
import { AQI_CATEGORIES, formatAQI, formatDateTime, getAQICategory } from '../../utils/aqiHelpers';

const METRICS = [
  { key: 'aqi', code: 'INDEX', label: 'AQI (Actual vs Forecast)', unit: '', color: '#14b8a6', decimals: 0 },
  { key: 'pm25', code: 'PM₂.₅', label: 'PM2.5', unit: ' µg/m³', color: '#f59e0b', decimals: 1 },
  { key: 'pm10', code: 'PM₁₀', label: 'PM10', unit: ' µg/m³', color: '#fb923c', decimals: 1 },
  { key: 'temperature', code: 'T_amb', label: 'Temp', unit: ' °C', color: '#ef4444', decimals: 1 },
  { key: 'relativeHumidity', code: 'RH', label: 'Humidity', unit: ' %', color: '#38bdf8', decimals: 1 },
  { key: 'pressure', code: 'P_atm', label: 'Pressure', unit: ' hPa', color: '#a855f7', decimals: 1 },
];

const RANGES = [
  { key: '24h', hours: 24, label: '24H' },
  { key: '48h', hours: 48, label: '48H (MODEL WINDOW)' },
  { key: '7d', hours: 168, label: '7D (168H)' },
];

const MODEL_TAGS = {
  'Best available': 'Auto-routed · GBDT 336-lag',
  'Tuned XGBoost': 'Optuna GBDT · 48×7 flattened',
  XGBoost: 'Gradient Boosted Trees (336-lag)',
  'Soft-Gated Ensemble': 'Learned Gate · AMSE Loss',
  'Two-Stage Regressor': 'Peak Classifier + Specialist',
  'CNN-LSTM Attention': '1D-Conv + Temporal Attention',
  'CNN-GRU Attention': '1D-Conv + GRU Attention',
  GRU: 'Recurrent Gated Unit',
  'Stacked LSTM': '2-Layer Recurrent Memory',
  'Base LSTM': 'Single-Layer Recurrent',
};

const AQI_Y_MAX = 340;
const PRED_COLOR = '#f59e0b';

function axisTick(ts, rangeKey) {
  const d = new Date(ts);
  if (rangeKey === '7d') {
    return d.toLocaleDateString('en-US', {
      weekday: 'short',
      hour: '2-digit',
      hour12: false,
    });
  }
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

function ChartTooltip({ active, payload, label, metricInfo, activeModelName }) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload || {};
  const val = row.value;
  const pred = row.predictedAqi;
  const isFuture = row.isFuturePoint;
  const cat = metricInfo.key === 'aqi' ? getAQICategory(val ?? pred) : null;
  const err =
    metricInfo.key === 'aqi' && val != null && pred != null ? Math.round(pred - val) : null;

  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-time mono">
        {formatDateTime(label)} {isFuture ? '· [NEXT HOUR t+1h]' : ''}
      </div>
      {val != null && (
        <div className="chart-tooltip-row">
          <span
            className="chart-tooltip-value mono"
            style={{ color: cat ? cat.color : metricInfo.color }}
          >
            {val}
            {metricInfo.unit}
          </span>
          <span className="chart-tooltip-label mono">
            {metricInfo.key === 'aqi' ? 'Actual AQI (OpenAQ)' : metricInfo.label}
          </span>
        </div>
      )}
      {metricInfo.key === 'aqi' && pred != null && (
        <>
          <div className="chart-tooltip-row" style={{ marginTop: val != null ? 4 : 0 }}>
            <span className="chart-tooltip-value mono" style={{ color: PRED_COLOR }}>
              {pred}
            </span>
            <span className="chart-tooltip-label mono">
              Predicted (t+1 · {activeModelName})
            </span>
          </div>
          {err != null && (
            <div className="chart-tooltip-cat mono" style={{ color: 'var(--text-secondary)' }}>
              Error (Pred − Actual): {err > 0 ? `+${err}` : err} AQI
            </div>
          )}
        </>
      )}
      {cat && (
        <div className="chart-tooltip-cat mono" style={{ color: cat.color }}>
          {cat.code} · {cat.shortLabel}
        </div>
      )}
    </div>
  );
}

/**
 * Unified AI Forecast (t+1h) & Historical Waveform Verification Studio.
 * Synchronizes a single model selector across:
 *   1. The t+1h point projection, trajectory & 90% Confidence Interval (left panel)
 *   2. The rolling 1-step-ahead Predicted vs. Actual OpenAQ waveform & live MAE/RMSE (right panel)
 */
export default function SensorChart({ data = [], currentAqi, loading, error }) {
  const [model, setModel] = useState('Best available');
  const [metric, setMetric] = useState('aqi');
  const [range, setRange] = useState('48h');

  const [forecast, setForecast] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(true);
  const [modelHistory, setModelHistory] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setForecastLoading(true);

    Promise.all([
      fetchForecast({ horizon: 1, model }),
      fetchHistoricalData(168, model),
    ])
      .then(([fcData, histData]) => {
        if (cancelled) return;
        setForecast(fcData);
        if (Array.isArray(histData) && histData.length > 0) {
          setModelHistory(histData);
        }
        setForecastLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setForecastLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [model, data]);

  const activeSeries = modelHistory && modelHistory.length > 0 ? modelHistory : data;
  const metricInfo = METRICS.find((m) => m.key === metric);
  const rangeInfo = RANGES.find((r) => r.key === range);

  const chartData = useMemo(() => {
    const sliced = activeSeries.slice(-rangeInfo.hours).map((d) => ({
      timestamp: d.timestamp,
      value: d[metric],
      predictedAqi: metric === 'aqi' ? d.predictedAqi ?? null : null,
      isFuturePoint: false,
    }));

    // Append the t+1h future projection at the tail of the AQI chart
    if (metric === 'aqi' && forecast?.predicted_aqi != null && sliced.length > 0) {
      const lastTs = new Date(sliced[sliced.length - 1].timestamp).getTime();
      if (!isNaN(lastTs)) {
        const nextTs = new Date(lastTs + 3600 * 1000).toISOString();
        sliced.push({
          timestamp: nextTs,
          value: null,
          predictedAqi: forecast.predicted_aqi,
          isFuturePoint: true,
        });
      }
    }
    return sliced;
  }, [activeSeries, metric, rangeInfo, forecast]);

  const stats = useMemo(() => {
    const observedRows = chartData.filter((d) => !d.isFuturePoint);
    const vals = observedRows.map((d) => d.value).filter((v) => v != null && !isNaN(v));
    if (!vals.length) return null;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const peakCount = vals.filter((v) => v >= 200).length;

    let liveMae = null;
    let liveRmse = null;
    if (metric === 'aqi') {
      const pairs = observedRows.filter(
        (d) => d.value != null && !isNaN(d.value) && d.predictedAqi != null && !isNaN(d.predictedAqi),
      );
      if (pairs.length > 0) {
        const absSum = pairs.reduce((acc, p) => acc + Math.abs(p.predictedAqi - p.value), 0);
        const sqSum = pairs.reduce((acc, p) => acc + (p.predictedAqi - p.value) ** 2, 0);
        liveMae = (absSum / pairs.length).toFixed(2);
        liveRmse = Math.sqrt(sqSum / pairs.length).toFixed(2);
      }
    }

    return {
      min: min.toFixed(metricInfo.decimals),
      max: max.toFixed(metricInfo.decimals),
      mean: mean.toFixed(metricInfo.decimals),
      peakCount,
      liveMae,
      liveRmse,
    };
  }, [chartData, metric, metricInfo]);

  const isAqi = metric === 'aqi';
  const resolvedCurrentAqi =
    currentAqi ?? (activeSeries.length ? activeSeries[activeSeries.length - 1]?.aqi : null);
  const category = forecast ? getAQICategory(forecast.predicted_aqi) : getAQICategory(resolvedCurrentAqi);
  const [lo, hi] = forecast?.confidence_interval ?? [0, 500];
  const delta =
    forecast && resolvedCurrentAqi != null
      ? Math.round(forecast.predicted_aqi - resolvedCurrentAqi)
      : null;
  const activeModelName = forecast?.model_used || (model === 'Best available' ? 'XGBoost' : model);

  return (
    <div className="forecast-studio">
      {/* LEFT PANEL: Next-Hour (t+1h) Point Forecast & Model Control */}
      <div className="forecast-studio-sidebar">
        <div className="forecast-controls">
          <span className="forecast-horizon-pill mono">HORIZON // t+1h</span>
          <select
            className="select mono"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            aria-label="Forecast model"
          >
            {FORECAST_MODELS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        {forecastLoading && !forecast ? (
          <div className="panel-loading mono">Computing t+1h projection…</div>
        ) : (
          forecast && (
            <>
              <div className="forecast-hero">
                <div className="forecast-hero-left">
                  <span className="forecast-kicker mono">PROJECTED INDEX // T+1H</span>
                  <div className="forecast-value-row">
                    <span className="forecast-value mono" style={{ color: category.color }}>
                      {formatAQI(forecast.predicted_aqi)}
                    </span>
                    {delta !== null && (
                      <span
                        className={`forecast-delta mono ${
                          delta > 0 ? 'is-rising' : delta < 0 ? 'is-falling' : 'is-flat'
                        }`}
                      >
                        {delta > 0 ? (
                          <FiArrowUpRight aria-hidden="true" />
                        ) : delta < 0 ? (
                          <FiArrowDownRight aria-hidden="true" />
                        ) : (
                          <FiArrowRight aria-hidden="true" />
                        )}
                        {delta > 0 ? `+${delta}` : delta} vs now
                      </span>
                    )}
                  </div>

                  <div
                    className="forecast-category"
                    style={{ background: category.bgColor, borderColor: category.borderColor }}
                  >
                    <span className="forecast-category-dot" style={{ background: category.color }} />
                    <span className="mono">{category.code}</span>
                    <span>·</span>
                    <span>{category.label}</span>
                  </div>
                </div>
              </div>

              <div className="forecast-model-box">
                <div className="forecast-model-header mono">
                  <FiCpu aria-hidden="true" />
                  <span>ACTIVE ENGINE (SYNCED WITH CHART)</span>
                </div>
                <div className="forecast-model-name">{activeModelName}</div>
                <div className="forecast-model-arch mono">
                  {MODEL_TAGS[model] || '48h × 7 sensor window'}
                </div>
              </div>

              {/* Trajectory comparison strip: NOW -> t+1h */}
              <div className="forecast-trajectory">
                <div className="trajectory-node">
                  <span className="trajectory-step mono">NOW (t₀)</span>
                  <span className="trajectory-val mono">
                    {resolvedCurrentAqi != null ? formatAQI(resolvedCurrentAqi) : '—'}
                  </span>
                </div>
                <span className="trajectory-arrow mono">→</span>
                <div className="trajectory-node is-selected">
                  <span className="trajectory-step mono">FORECAST (t + 1h)</span>
                  <span className="trajectory-val mono" style={{ color: category.color }}>
                    {formatAQI(forecast.predicted_aqi)}
                  </span>
                </div>
              </div>

              {/* 90% Prediction Interval summary cells */}
              <div className="forecast-ci-grid">
                <div className="forecast-ci-cell">
                  <span className="forecast-ci-cell-label mono">LOWER 90% CI</span>
                  <span className="forecast-ci-cell-val mono">{lo} AQI</span>
                </div>
                <div className="forecast-ci-cell">
                  <span className="forecast-ci-cell-label mono">UPPER 90% CI</span>
                  <span className="forecast-ci-cell-val mono">{hi} AQI</span>
                </div>
                <div className="forecast-ci-cell">
                  <span className="forecast-ci-cell-label mono">MARGIN</span>
                  <span className="forecast-ci-cell-val mono">±{Math.round((hi - lo) / 2)}</span>
                </div>
              </div>
            </>
          )
        )}
      </div>

      {/* RIGHT PANEL: Historical Waveform & Live Verification */}
      <div className="forecast-studio-main">
        <div className="chart-controls">
          <div className="segmented" role="tablist" aria-label="Metric">
            {METRICS.map((m) => (
              <button
                key={m.key}
                type="button"
                className={metric === m.key ? 'active' : ''}
                onClick={() => setMetric(m.key)}
              >
                <span className="segmented-code mono">{m.code}</span>
                <span>{m.label}</span>
              </button>
            ))}
          </div>
          <div className="segmented" role="tablist" aria-label="Time range">
            {RANGES.map((r) => (
              <button
                key={r.key}
                type="button"
                className={`mono ${range === r.key ? 'active' : ''}`}
                onClick={() => setRange(r.key)}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>

        {stats && (
          <div className="chart-stats-ribbon">
            <div className="chart-stat-pill">
              <span className="chart-stat-label mono">WINDOW MEAN</span>
              <span className="chart-stat-val mono">
                {stats.mean}
                {metricInfo.unit}
              </span>
            </div>
            <div className="chart-stat-pill">
              <span className="chart-stat-label mono">MIN / MAX</span>
              <span className="chart-stat-val mono" style={{ color: metricInfo.color }}>
                {stats.min} / {stats.max}
                {metricInfo.unit}
              </span>
            </div>
            {isAqi && stats.liveMae != null && (
              <div className="chart-stat-pill">
                <span className="chart-stat-label mono">LIVE MAE ({activeModelName})</span>
                <span className="chart-stat-val mono" style={{ color: PRED_COLOR }}>
                  {stats.liveMae} AQI
                </span>
              </div>
            )}
            {isAqi && stats.liveRmse != null && (
              <div className="chart-stat-pill">
                <span className="chart-stat-label mono">LIVE RMSE ({activeModelName})</span>
                <span className="chart-stat-val mono" style={{ color: PRED_COLOR }}>
                  {stats.liveRmse} AQI
                </span>
              </div>
            )}
            {isAqi && (
              <div className="chart-stat-pill">
                <span className="chart-stat-label mono">SERIES</span>
                <span className="chart-stat-val mono" style={{ fontSize: '0.76rem' }}>
                  <span style={{ color: '#14b8a6' }}>━ Actual (OpenAQ)</span>
                  {'  '}
                  <span style={{ color: PRED_COLOR }}>┅ {activeModelName} (t+1)</span>
                </span>
              </div>
            )}
          </div>
        )}

        {loading && <div className="panel-loading mono">Streaming waveform buffer…</div>}
        {!loading && error && <div className="panel-error">{error}</div>}
        {!loading && !error && (
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={chartData} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={`chartFill-${metric}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={metricInfo.color} stopOpacity={0.28} />
                  <stop offset="85%" stopColor={metricInfo.color} stopOpacity={0.02} />
                  <stop offset="100%" stopColor={metricInfo.color} stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(ts) => axisTick(ts, range)}
                tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                axisLine={{ stroke: 'var(--chart-axis)' }}
                tickLine={false}
                minTickGap={48}
              />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                axisLine={false}
                tickLine={false}
                width={44}
                domain={isAqi ? [0, AQI_Y_MAX] : ['auto', 'auto']}
              />

              {isAqi &&
                AQI_CATEGORIES.map((cat) => (
                  <ReferenceArea
                    key={cat.level}
                    y1={Math.min(cat.min, AQI_Y_MAX)}
                    y2={Math.min(cat.max, AQI_Y_MAX)}
                    fill={cat.color}
                    fillOpacity={0.045}
                    stroke="none"
                  />
                ))}
              {isAqi && (
                <ReferenceLine
                  y={200}
                  stroke="var(--text-muted)"
                  strokeWidth={1}
                  strokeDasharray="5 4"
                  strokeOpacity={0.6}
                  label={{
                    value: 'PEAK REGIME ≥ 200',
                    position: 'insideTopRight',
                    fill: 'var(--text-secondary)',
                    fontSize: 10,
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              )}

              <Tooltip
                content={
                  <ChartTooltip metricInfo={metricInfo} activeModelName={activeModelName} />
                }
                cursor={{ stroke: 'var(--border-strong)', strokeWidth: 1, strokeDasharray: '3 3' }}
              />
              <Area
                type="monotone"
                dataKey="value"
                name="Actual AQI (OpenAQ)"
                stroke={metricInfo.color}
                strokeWidth={2.25}
                fill={`url(#chartFill-${metric})`}
                dot={false}
                connectNulls={false}
                activeDot={{
                  r: 5,
                  fill: metricInfo.color,
                  stroke: 'var(--bg-primary)',
                  strokeWidth: 2,
                }}
                isAnimationActive={false}
              />
              {isAqi && (
                <Line
                  type="monotone"
                  dataKey="predictedAqi"
                  name={`Predicted AQI (t+1 · ${activeModelName})`}
                  stroke={PRED_COLOR}
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  dot={false}
                  activeDot={{
                    r: 4,
                    fill: PRED_COLOR,
                    stroke: 'var(--bg-primary)',
                    strokeWidth: 2,
                  }}
                  isAnimationActive={false}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
