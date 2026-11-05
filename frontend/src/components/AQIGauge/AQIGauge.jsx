import { AQI_CATEGORIES, formatAQI, formatSensorValue, getAQICategory } from '../../utils/aqiHelpers';

const W = 280;
const H = 160;
const CX = 140;
const CY = 136;
const R = 102;
const STROKE = 14;
const MAX_AQI = 500;
const TICKS = [0, 50, 100, 150, 200, 300, 500];

/** Point on a semicircle of radius `r` for fraction t in [0, 1] (0 = west, 1 = east). */
function pointAt(t, r = R) {
  const theta = t * Math.PI;
  return {
    x: CX - r * Math.cos(theta),
    y: CY - r * Math.sin(theta),
  };
}

/** SVG arc path for a [t0, t1] slice of the semicircle (clockwise). */
function segmentPath(t0, t1, r = R) {
  const p0 = pointAt(t0, r);
  const p1 = pointAt(t1, r);
  const largeArc = t1 - t0 > 0.5 ? 1 : 0;
  return `M ${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A ${r} ${r} 0 ${largeArc} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`;
}

/**
 * Precision meteorological AQI dial with US-EPA color bands, scale ticks,
 * and 1h/24h telemetry drift cells.
 */
export default function AQIGauge({ value, current, historical = [] }) {
  const category = getAQICategory(value);
  const t = Math.min(1, Math.max(0, (value ?? 0) / MAX_AQI));
  const pointer = pointAt(t, R);

  const prev1h = historical.length >= 2 ? historical[historical.length - 2]?.aqi : null;
  const prev24h = historical.length >= 25 ? historical[historical.length - 25]?.aqi : null;
  const delta1h = value != null && prev1h != null ? Math.round(value - prev1h) : null;
  const delta24h = value != null && prev24h != null ? Math.round(value - prev24h) : null;

  return (
    <div className="gauge">
      <div className="gauge-top-meta">
        <div className="gauge-badge" style={{ background: category.bgColor, borderColor: category.borderColor }}>
          <span className="gauge-category-dot" style={{ background: category.color }} />
          <span className="mono">{category.code}</span>
          <span>·</span>
          <span>{category.label}</span>
        </div>
      </div>

      <div className="gauge-wrap">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Current AQI ${formatAQI(value)}, ${category.label}`}
        >
          {/* Segmented US-EPA color bands */}
          <path
            d={segmentPath(0, 1, R)}
            fill="none"
            stroke="var(--gauge-track)"
            strokeWidth={STROKE}
            strokeLinecap="butt"
          />
          {AQI_CATEGORIES.map((cat) => {
            const t0 = Math.min(1, cat.min / MAX_AQI);
            const t1 = Math.min(1, (cat.max === Infinity ? MAX_AQI : cat.max) / MAX_AQI);
            return (
              <path
                key={cat.level}
                d={segmentPath(t0, t1, R)}
                fill="none"
                stroke={cat.color}
                strokeOpacity={0.85}
                strokeWidth={STROKE}
                strokeLinecap="butt"
              />
            );
          })}

          {/* Scale tick marks & labels */}
          {TICKS.map((tickVal) => {
            const tickT = tickVal / MAX_AQI;
            const pOuter = pointAt(tickT, R + STROKE / 2 + 2);
            const pInner = pointAt(tickT, R - STROKE / 2 - 2);
            const pLabel = pointAt(tickT, R + STROKE / 2 + 13);
            return (
              <g key={tickVal}>
                <line
                  x1={pInner.x}
                  y1={pInner.y}
                  x2={pOuter.x}
                  y2={pOuter.y}
                  stroke="var(--bg-primary)"
                  strokeWidth={2}
                />
                <text
                  x={pLabel.x}
                  y={pLabel.y}
                  fill="var(--text-muted)"
                  fontSize="8.5"
                  fontFamily="var(--font-mono)"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {tickVal}
                </text>
              </g>
            );
          })}

          {/* Active pointer on the arc */}
          <circle
            cx={pointer.x}
            cy={pointer.y}
            r={7}
            fill="var(--bg-primary)"
            stroke={category.color}
            strokeWidth={3.5}
          />
        </svg>

        <div className="gauge-center">
          <span className="gauge-unit mono">US-EPA INDEX</span>
          <span className="gauge-value mono" style={{ color: category.color }}>
            {formatAQI(value)}
          </span>
          <span className="gauge-dominant mono">
            DOMINANT: PM2.5 ({current ? `${formatSensorValue(current.pm25, 1)} µg/m³` : '—'})
          </span>
        </div>
      </div>

      {/* Telemetry delta strip */}
      <div className="gauge-deltas">
        <div className="gauge-delta-cell">
          <span className="gauge-delta-label mono">1H DRIFT</span>
          <span
            className={`gauge-delta-val mono ${
              delta1h > 0 ? 'is-rising' : delta1h < 0 ? 'is-falling' : ''
            }`}
          >
            {delta1h == null ? '—' : `${delta1h > 0 ? '+' : ''}${delta1h} AQI`}
          </span>
        </div>
        <div className="gauge-delta-cell">
          <span className="gauge-delta-label mono">24H DRIFT</span>
          <span
            className={`gauge-delta-val mono ${
              delta24h > 0 ? 'is-rising' : delta24h < 0 ? 'is-falling' : ''
            }`}
          >
            {delta24h == null ? '—' : `${delta24h > 0 ? '+' : ''}${delta24h} AQI`}
          </span>
        </div>
        <div className="gauge-delta-cell">
          <span className="gauge-delta-label mono">REGIME</span>
          <span className="gauge-delta-val mono">
            {value >= 200 ? 'PEAK (≥200)' : 'NOMINAL (<200)'}
          </span>
        </div>
      </div>

      <p className="gauge-advice">{category.advice}</p>
    </div>
  );
}
