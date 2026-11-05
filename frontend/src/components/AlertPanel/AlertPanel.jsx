import { useMemo, useState } from 'react';
import { FiAlertOctagon, FiAlertTriangle, FiInfo } from 'react-icons/fi';

const SEVERITY = {
  hazardous: {
    code: 'ALT-301',
    label: 'Hazardous Spike',
    color: '#f43f5e',
    bg: 'rgba(244, 63, 94, 0.14)',
    border: 'rgba(244, 63, 94, 0.35)',
    icon: FiAlertOctagon,
    tier: 'critical',
  },
  very_unhealthy: {
    code: 'ALT-201',
    label: 'Very Unhealthy',
    color: '#c084fc',
    bg: 'rgba(192, 132, 252, 0.14)',
    border: 'rgba(192, 132, 252, 0.35)',
    icon: FiAlertTriangle,
    tier: 'critical',
  },
  forecast_high: {
    code: 'PRJ-200',
    label: 'Projected Peak ≥ 200',
    color: '#f59e0b',
    bg: 'rgba(245, 158, 11, 0.14)',
    border: 'rgba(245, 158, 11, 0.35)',
    icon: FiInfo,
    tier: 'warning',
  },
};

const FILTERS = [
  { key: 'all', label: 'All Events' },
  { key: 'critical', label: 'Critical (Observed)' },
  { key: 'warning', label: 'Projected Warnings' },
];

function relativeTime(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${Math.max(1, mins)}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/**
 * Operational anomaly & alert log with severity filters, incident codes,
 * and non-color-exclusive icons/badges.
 */
export default function AlertPanel({ alerts, loading, error, limit }) {
  const [filter, setFilter] = useState('all');

  const rows = useMemo(() => {
    const filtered =
      filter === 'all'
        ? alerts
        : alerts.filter((a) => SEVERITY[a.type]?.tier === filter);
    return limit ? filtered.slice(0, limit) : filtered;
  }, [alerts, filter, limit]);

  const criticalCount = alerts.filter((a) => SEVERITY[a.type]?.tier === 'critical').length;

  if (loading) return <div className="panel-loading mono">Scanning anomaly log…</div>;
  if (error) return <div className="panel-error">{error}</div>;

  return (
    <div className="alert-panel">
      <div className="alert-controls">
        <div className="segmented" role="tablist" aria-label="Alert severity">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              className={filter === f.key ? 'active' : ''}
              onClick={() => setFilter(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>
        <span className="alert-summary mono">
          {criticalCount} CRITICAL · {alerts.length - criticalCount} PROJECTED
        </span>
      </div>

      {rows.length === 0 ? (
        <div className="panel-empty mono">No anomaly events match the active filter.</div>
      ) : (
        <ul className="alert-list">
          {rows.map((a) => {
            const sev = SEVERITY[a.type] ?? SEVERITY.forecast_high;
            const Icon = sev.icon;
            return (
              <li key={a.id} className="alert-item">
                <span
                  className="alert-icon"
                  style={{ color: sev.color, background: sev.bg, borderColor: sev.border }}
                >
                  <Icon aria-hidden="true" />
                </span>
                <div className="alert-body">
                  <div className="alert-title-row">
                    <span className="alert-code mono" style={{ color: sev.color }}>
                      {sev.code}
                    </span>
                    <span className="alert-title">{a.message}</span>
                  </div>
                  <div className="alert-meta">
                    <span
                      className="alert-badge mono"
                      style={{ color: sev.color, background: sev.bg, borderColor: sev.border }}
                    >
                      {sev.label}
                    </span>
                    <span className="alert-time mono">T-{relativeTime(a.timestamp)}</span>
                  </div>
                </div>
                <div className="alert-aqi-box">
                  <span className="alert-aqi-label mono">PEAK INDEX</span>
                  <span className="alert-aqi mono" style={{ color: sev.color }}>
                    AQI {a.aqiValue}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
