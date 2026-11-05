import { FiActivity, FiCloud, FiDroplet, FiLayers, FiThermometer } from 'react-icons/fi';
import { formatSensorValue } from '../../utils/aqiHelpers';

const METRICS = [
  {
    key: 'pm25',
    code: 'PM₂.₅',
    label: 'Fine Particulate',
    unit: 'µg/m³',
    icon: FiCloud,
    decimals: 1,
    refLabel: 'WHO 24h: 15 µg/m³',
    accent: '#f59e0b',
  },
  {
    key: 'pm10',
    code: 'PM₁₀',
    label: 'Coarse Particulate',
    unit: 'µg/m³',
    icon: FiLayers,
    decimals: 1,
    refLabel: 'WHO 24h: 45 µg/m³',
    accent: '#fb923c',
  },
  {
    key: 'temperature',
    code: 'T_amb',
    label: 'Ambient Temp',
    unit: '°C',
    icon: FiThermometer,
    decimals: 1,
    refLabel: 'Diurnal thermal cycle',
    accent: '#ef4444',
  },
  {
    key: 'relativeHumidity',
    code: 'RH',
    label: 'Relative Humidity',
    unit: '%',
    icon: FiDroplet,
    decimals: 1,
    refLabel: 'Hygroscopic growth factor',
    accent: '#38bdf8',
  },
  {
    key: 'pressure',
    code: 'P_atm',
    label: 'Barometric Press.',
    unit: 'hPa',
    icon: FiActivity,
    decimals: 1,
    refLabel: 'Boundary layer inversion',
    accent: '#14b8a6',
  },
];

/** Build an SVG polyline string for a 24h mini sparkline. */
function buildSparkline(values, width = 120, height = 28) {
  if (!values.length) return '';
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values
    .map((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * width;
      const y = height - 3 - ((v - min) / span) * (height - 6);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

/**
 * 5-Channel IoT Sensor Telemetry Strip with 24h mini-sparklines,
 * min/max range bars, and physical channel symbols.
 */
export default function SensorReadings({ current, historical = [] }) {
  const recent24 = historical.slice(-24);

  return (
    <div className="stat-tiles">
      {METRICS.map(({ key, code, label, unit, icon: Icon, decimals, refLabel, accent }) => {
        const val = current?.[key];
        const series = recent24.map((d) => d[key]).filter((v) => v != null && !isNaN(v));
        const min = series.length ? Math.min(...series) : null;
        const max = series.length ? Math.max(...series) : null;
        const sparkPoints = buildSparkline(series);
        const pct =
          val != null && min != null && max != null && max > min
            ? Math.min(100, Math.max(0, ((val - min) / (max - min)) * 100))
            : 50;

        return (
          <div className="stat-tile" key={key} style={{ '--tile-accent': accent }}>
            <div className="stat-tile-top">
              <span className="stat-tile-code mono">{code}</span>
              <span className="stat-tile-icon">
                <Icon aria-hidden="true" />
              </span>
            </div>

            <span className="stat-tile-label">{label}</span>

            <div className="stat-tile-value-row">
              <span className="stat-tile-value mono">
                {current ? formatSensorValue(val, decimals) : '—'}
              </span>
              <span className="stat-tile-unit mono">{unit}</span>
            </div>

            {sparkPoints && (
              <div className="stat-tile-spark" aria-hidden="true">
                <svg viewBox="0 0 120 28" preserveAspectRatio="none">
                  <polyline
                    fill="none"
                    stroke={accent}
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={sparkPoints}
                  />
                </svg>
              </div>
            )}

            {min != null && max != null && (
              <div className="stat-tile-range">
                <div className="stat-tile-range-labels mono">
                  <span>L: {formatSensorValue(min, decimals)}</span>
                  <span>24H</span>
                  <span>H: {formatSensorValue(max, decimals)}</span>
                </div>
                <div className="stat-tile-range-track">
                  <span
                    className="stat-tile-range-fill"
                    style={{ width: `${pct}%`, background: accent }}
                  />
                </div>
              </div>
            )}

            <span className="stat-tile-ref mono">{refLabel}</span>
          </div>
        );
      })}
    </div>
  );
}
