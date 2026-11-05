import { useEffect, useState } from 'react';
import { FiArrowDownRight, FiArrowRight, FiArrowUpRight, FiCpu } from 'react-icons/fi';
import { FORECAST_MODELS, fetchForecast } from '../../api/aqiApi';
import { formatAQI, getAQICategory } from '../../utils/aqiHelpers';

const MODEL_TAGS = {
  'Best available': 'Auto-routed · GBDT 336-lag',
  'Tuned XGBoost': 'Optuna GBDT · 48×7 flattened',
  XGBoost: 'Gradient Boosted Trees',
  'Soft-Gated Ensemble': 'Learned Gate · AMSE Loss',
  'Two-Stage Regressor': 'Peak Classifier + Specialist',
  'CNN-LSTM Attention': '1D-Conv + Temporal Attention',
  'CNN-GRU Attention': '1D-Conv + GRU Attention',
  GRU: 'Recurrent Gated Unit',
  'Stacked LSTM': '2-Layer Recurrent Memory',
  'Base LSTM': 'Single-Layer Recurrent',
};

/**
 * Self-contained t+1h forecast module.
 * Forecasts 1 hour ahead using the selected model architecture.
 */
export default function ForecastCard({ currentAqi }) {
  const [model, setModel] = useState('Best available');
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchForecast({ horizon: 1, model })
      .then((data) => {
        if (cancelled) return;
        setForecast(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err?.message || 'Forecast unavailable');
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [model, currentAqi]);

  const category = forecast ? getAQICategory(forecast.predicted_aqi) : null;
  const [lo, hi] = forecast?.confidence_interval ?? [0, 500];
  const delta =
    forecast && currentAqi != null ? Math.round(forecast.predicted_aqi - currentAqi) : null;

  return (
    <div className="forecast">
      <div className="forecast-controls">
        <span className="forecast-horizon-pill mono">HORIZON // t+1h (1-HOUR AHEAD)</span>
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

      {loading && <div className="panel-loading mono">Computing t+1h projection…</div>}
      {!loading && error && <div className="panel-error">{error}</div>}

      {!loading && !error && forecast && (
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

            <div className="forecast-model-box">
              <div className="forecast-model-header mono">
                <FiCpu aria-hidden="true" />
                <span>ACTIVE ENGINE</span>
              </div>
              <div className="forecast-model-name">{forecast.model_used}</div>
              <div className="forecast-model-arch mono">
                {MODEL_TAGS[model] || '48h × 7 sensor window'}
              </div>
            </div>
          </div>

          {/* Trajectory comparison strip: NOW -> t+1h */}
          <div className="forecast-trajectory">
            <div className="trajectory-node">
              <span className="trajectory-step mono">NOW (t₀)</span>
              <span className="trajectory-val mono">
                {currentAqi != null ? formatAQI(currentAqi) : '—'}
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

          {/* 90% Prediction Interval summary cells (no purple bars) */}
          <div className="forecast-ci-grid">
            <div className="forecast-ci-cell">
              <span className="forecast-ci-cell-label mono">LOWER BOUND (90% CI)</span>
              <span className="forecast-ci-cell-val mono">{lo} AQI</span>
            </div>
            <div className="forecast-ci-cell">
              <span className="forecast-ci-cell-label mono">UPPER BOUND (90% CI)</span>
              <span className="forecast-ci-cell-val mono">{hi} AQI</span>
            </div>
            <div className="forecast-ci-cell">
              <span className="forecast-ci-cell-label mono">UNCERTAINTY MARGIN</span>
              <span className="forecast-ci-cell-val mono">±{Math.round((hi - lo) / 2)} AQI</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
