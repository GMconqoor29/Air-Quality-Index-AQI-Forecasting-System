import { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { FiAward, FiCheck, FiSliders, FiZap } from 'react-icons/fi';

const METRICS = [
  { key: 'rmse', label: 'RMSE', desc: 'Root Mean Squared Error', decimals: 2, higherIsBetter: false },
  { key: 'mae', label: 'MAE', desc: 'Mean Absolute Error', decimals: 2, higherIsBetter: false },
  {
    key: 'peakRecall',
    label: 'Peak Recall (≥200)',
    desc: 'Capture rate of hazardous spikes',
    decimals: 1,
    higherIsBetter: true,
    ratio: true,
  },
  {
    key: 'peakMAE',
    label: 'Peak MAE',
    desc: 'Error during high-AQI events',
    decimals: 2,
    higherIsBetter: false,
  },
  {
    key: 'amse',
    label: 'AMSE',
    desc: 'Asymmetric Penalty MSE',
    decimals: 1,
    higherIsBetter: false,
  },
];

const MODEL_FAMILIES = {
  Persistence: { tag: 'BASELINE', color: '#64748b' },
  'Base LSTM': { tag: 'RECURRENT', color: '#38bdf8' },
  GRU: { tag: 'RECURRENT', color: '#38bdf8' },
  'Stacked LSTM': { tag: 'RECURRENT', color: '#38bdf8' },
  XGBoost: { tag: 'GBDT (336-LAG)', color: '#10b981' },
  'Tuned XGBoost': { tag: 'OPTUNA GBDT', color: '#10b981' },
  'Soft-Gated Ensemble': { tag: 'AMSE ENSEMBLE', color: '#f59e0b' },
  'CNN-LSTM Attention': { tag: 'CONV-ATTENTION', color: '#a855f7' },
  'CNN-GRU Attention': { tag: 'CONV-ATTENTION', color: '#a855f7' },
  'Two-Stage Regressor': { tag: 'PEAK SPECIALIST', color: '#f43f5e' },
};

function formatValue(metric, value) {
  if (metric.ratio) return `${(value * 100).toFixed(metric.decimals)}%`;
  return value.toFixed(metric.decimals);
}

function BarTooltip({ active, payload, metric }) {
  if (!active || !payload?.length) return null;
  const row = payload[0];
  const fam = MODEL_FAMILIES[row.payload.model];
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-time mono">
        {row.payload.model} {fam ? `// ${fam.tag}` : ''}
      </div>
      <div className="chart-tooltip-row">
        <span className="chart-tooltip-value mono">{formatValue(metric, row.value)}</span>
        <span className="chart-tooltip-label mono">{metric.label}</span>
      </div>
    </div>
  );
}

/**
 * Model benchmark leaderboard with regime trade-off callouts,
 * architecture family tags, and proportional inline telemetry bars.
 */
export default function ModelComparison({ data, loading, error }) {
  const [focus, setFocus] = useState('rmse');

  const focusMetric = METRICS.find((m) => m.key === focus);

  const sorted = useMemo(() => {
    const copy = [...data];
    copy.sort((a, b) =>
      focusMetric.higherIsBetter ? b[focus] - a[focus] : a[focus] - b[focus],
    );
    return copy;
  }, [data, focus, focusMetric]);

  const metricBounds = useMemo(() => {
    const bounds = {};
    METRICS.forEach((m) => {
      const vals = data.map((d) => d[m.key]);
      bounds[m.key] = {
        min: Math.min(...vals),
        max: Math.max(...vals),
        best: m.higherIsBetter ? Math.max(...vals) : Math.min(...vals),
      };
    });
    return bounds;
  }, [data]);

  if (loading) return <div className="panel-loading mono">Loading holdout benchmark matrix…</div>;
  if (error) return <div className="panel-error">{error}</div>;
  if (!sorted.length) return <div className="panel-empty">No model results yet.</div>;

  return (
    <div className="model-comparison">
      {/* Scientific Trade-off Spotlights */}
      <div className="benchmark-spotlights">
        <div className="spotlight-card">
          <div className="spotlight-kicker mono">
            <FiAward aria-hidden="true" /> OVERALL POINT ACCURACY
          </div>
          <div className="spotlight-title">XGBoost (336-Lag)</div>
          <div className="spotlight-metric mono">28.42 RMSE · 19.66 MAE</div>
          <p className="spotlight-desc">
            Gradient boosted trees over flattened 48×7 sensor windows achieve the lowest aggregate
            error in nominal regimes.
          </p>
        </div>

        <div className="spotlight-card">
          <div className="spotlight-kicker mono">
            <FiZap aria-hidden="true" /> HIGH-AQI PEAK CAPTURE (≥200)
          </div>
          <div className="spotlight-title">Two-Stage Specialist</div>
          <div className="spotlight-metric mono">90.4% Peak Recall · 21.68 Peak MAE</div>
          <p className="spotlight-desc">
            Decouples regime classification from specialist peak regression to eliminate severe
            under-prediction during pollution spikes.
          </p>
        </div>

        <div className="spotlight-card">
          <div className="spotlight-kicker mono">
            <FiSliders aria-hidden="true" /> ASYMMETRIC HEALTH COST
          </div>
          <div className="spotlight-title">Soft-Gated Ensemble</div>
          <div className="spotlight-metric mono">1664.3 AMSE · 83.4% Recall</div>
          <p className="spotlight-desc">
            Learned gating network trained with AQI-weighted Asymmetric MSE balances routine
            precision with peak sensitivity.
          </p>
        </div>
      </div>

      <div className="chart-controls">
        <div className="segmented" role="tablist" aria-label="Comparison metric">
          {METRICS.map((m) => (
            <button
              key={m.key}
              type="button"
              className={focus === m.key ? 'active' : ''}
              onClick={() => setFocus(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <span className="model-focus-desc mono">
          SORTED BY: {focusMetric.desc.toUpperCase()} ({focusMetric.higherIsBetter ? 'HIGHER IS BETTER' : 'LOWER IS BETTER'})
        </span>
      </div>

      <div className="model-chart">
        <ResponsiveContainer width="100%" height={Math.max(240, sorted.length * 34)}>
          <BarChart
            data={sorted}
            layout="vertical"
            margin={{ top: 4, right: 32, bottom: 4, left: 8 }}
          >
            <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
              tickFormatter={(v) => (focusMetric.ratio ? `${Math.round(v * 100)}%` : Math.round(v))}
              axisLine={{ stroke: 'var(--chart-axis)' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="model"
              width={158}
              tick={{ fill: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              content={<BarTooltip metric={focusMetric} />}
              cursor={{ fill: 'var(--bg-secondary)' }}
            />
            <Bar dataKey={focus} barSize={16} radius={[0, 4, 4, 0]} isAnimationActive={false}>
              {sorted.map((entry, index) => (
                <Cell
                  key={entry.model}
                  fill={
                    index === 0
                      ? 'var(--accent)'
                      : MODEL_FAMILIES[entry.model]?.color || 'var(--accent)'
                  }
                  fillOpacity={index === 0 ? 1 : 0.65}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="model-table-wrap">
        <table className="model-table">
          <thead>
            <tr>
              <th className="model-table-rank mono">#</th>
              <th className="model-table-model">Architecture</th>
              <th className="model-table-family mono">Paradigm</th>
              {METRICS.map((m) => (
                <th
                  key={m.key}
                  className={`num sortable ${focus === m.key ? 'is-focus' : ''}`}
                  onClick={() => setFocus(m.key)}
                  aria-sort={focus === m.key ? 'ascending' : 'none'}
                >
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row, idx) => {
              const isBest = idx === 0;
              const fam = MODEL_FAMILIES[row.model] || { tag: 'MODEL', color: '#64748b' };
              return (
                <tr key={row.model} className={isBest ? 'is-best' : ''}>
                  <td className="model-table-rank mono">{String(idx + 1).padStart(2, '0')}</td>
                  <td className="model-table-model">
                    {isBest && (
                      <span className="best-badge" title={`Best ${focusMetric.label}`}>
                        <FiCheck aria-hidden="true" />
                      </span>
                    )}
                    {row.model}
                  </td>
                  <td>
                    <span
                      className="family-tag mono"
                      style={{
                        color: fam.color,
                        borderColor: `${fam.color}44`,
                        background: `${fam.color}14`,
                      }}
                    >
                      {fam.tag}
                    </span>
                  </td>
                  {METRICS.map((m) => {
                    const b = metricBounds[m.key];
                    const isColBest = row[m.key] === b.best;
                    return (
                      <td
                        key={m.key}
                        className={`num mono ${isColBest ? 'is-best-value' : ''}`}
                      >
                        {formatValue(m, row[m.key])}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="model-note mono">
        NOTE // Holdout evaluation on chronological split. Lower is better for RMSE, MAE, Peak MAE,
        and AMSE; higher is better for Peak Recall (AQI ≥ 200).
      </p>
    </div>
  );
}
