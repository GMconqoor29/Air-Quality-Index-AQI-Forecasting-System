/**
 * AQI Helper Utilities
 * Provides color coding, category labels, and health advice
 * based on the US EPA AQI standard.
 */

/**
 * AQI breakpoint definitions (US EPA Standard).
 */
const AQI_CATEGORIES = [
  {
    level: 0,
    code: 'EPA-I',
    shortLabel: 'Good',
    label: 'Good',
    range: '0–50',
    color: '#10b981',
    bgColor: 'rgba(16, 185, 129, 0.14)',
    borderColor: 'rgba(16, 185, 129, 0.35)',
    textColor: '#047857',
    darkTextColor: '#34d399',
    description: 'Air quality is satisfactory, and air pollution poses little or no risk.',
    advice: 'Air quality is clean and favorable. Ideal conditions for outdoor exercise and ventilation.',
    min: 0,
    max: 50,
  },
  {
    level: 1,
    code: 'EPA-II',
    shortLabel: 'Moderate',
    label: 'Moderate',
    range: '51–100',
    color: '#eab308',
    bgColor: 'rgba(234, 179, 8, 0.14)',
    borderColor: 'rgba(234, 179, 8, 0.36)',
    textColor: '#a16207',
    darkTextColor: '#facc15',
    description: 'Air quality is acceptable. Some pollutants may pose a moderate health concern for a small number of people.',
    advice: 'Unusually sensitive individuals should monitor symptoms during prolonged outdoor exertion.',
    min: 51,
    max: 100,
  },
  {
    level: 2,
    code: 'EPA-III',
    shortLabel: 'Sensitive Groups',
    label: 'Unhealthy for Sensitive Groups',
    range: '101–150',
    color: '#f97316',
    bgColor: 'rgba(249, 115, 22, 0.15)',
    borderColor: 'rgba(249, 115, 22, 0.38)',
    textColor: '#c2410c',
    darkTextColor: '#fb923c',
    description: 'Members of sensitive groups may experience health effects.',
    advice: 'Children, older adults, and people with respiratory conditions should limit prolonged outdoor exertion.',
    min: 101,
    max: 150,
  },
  {
    level: 3,
    code: 'EPA-IV',
    shortLabel: 'Unhealthy',
    label: 'Unhealthy',
    range: '151–200',
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.15)',
    borderColor: 'rgba(239, 68, 68, 0.40)',
    textColor: '#b91c1c',
    darkTextColor: '#f87171',
    description: 'Some members of the general public may experience health effects.',
    advice: 'Reduce prolonged or heavy outdoor exertion. Sensitive groups should move activities indoors.',
    min: 151,
    max: 200,
  },
  {
    level: 4,
    code: 'EPA-V',
    shortLabel: 'Very Unhealthy',
    label: 'Very Unhealthy',
    range: '201–300',
    color: '#a855f7',
    bgColor: 'rgba(168, 85, 247, 0.16)',
    borderColor: 'rgba(168, 85, 247, 0.42)',
    textColor: '#7e22ce',
    darkTextColor: '#c084fc',
    description: 'Health alert: Everyone may experience more serious health effects.',
    advice: 'Peak pollution regime (≥ 200). Avoid outdoor physical activity; keep windows closed and run HEPA filtration.',
    min: 201,
    max: 300,
  },
  {
    level: 5,
    code: 'EPA-VI',
    shortLabel: 'Hazardous',
    label: 'Hazardous',
    range: '301+',
    color: '#e11d48',
    bgColor: 'rgba(225, 29, 72, 0.18)',
    borderColor: 'rgba(225, 29, 72, 0.45)',
    textColor: '#9f1239',
    darkTextColor: '#fb7185',
    description: 'Health warning of emergency conditions: everyone is more likely to be affected.',
    advice: 'Emergency particulate levels. Remain indoors with sealed ventilation and wear N95/FFP2 protection if transit is unavoidable.',
    min: 301,
    max: Infinity,
  },
];

/**
 * Get full AQI category info for a given AQI value.
 * @param {number} aqi
 * @returns {{ level, label, color, bgColor, textColor, darkTextColor, description, range }}
 */
export function getAQICategory(aqi) {
  if (aqi == null || isNaN(aqi)) {
    return AQI_CATEGORIES[0];
  }
  const val = Math.max(0, Math.round(aqi));
  for (const cat of AQI_CATEGORIES) {
    if (val >= cat.min && val <= cat.max) {
      return cat;
    }
  }
  return AQI_CATEGORIES[AQI_CATEGORIES.length - 1];
}

/**
 * Get AQI level number (0–5).
 * @param {number} aqi
 * @returns {number}
 */
export function getAQILevel(aqi) {
  return getAQICategory(aqi).level;
}

/**
 * Round and format AQI as a string (no decimals).
 * @param {number} aqi
 * @returns {string}
 */
export function formatAQI(aqi) {
  if (aqi == null || isNaN(aqi)) return '—';
  return Math.round(aqi).toString();
}

/**
 * Get health advice text for a given AQI.
 * @param {number} aqi
 * @returns {string}
 */
export function getHealthAdvice(aqi) {
  return getAQICategory(aqi).advice;
}

/**
 * Calculate a color interpolated along the AQI spectrum.
 * Useful for smooth gradients.
 * @param {number} aqi
 * @returns {string} hex color
 */
export function getAQIColor(aqi) {
  return getAQICategory(aqi).color;
}

/**
 * Format a sensor reading value with appropriate decimal places.
 * @param {number} value
 * @param {number} decimals
 * @returns {string}
 */
export function formatSensorValue(value, decimals = 1) {
  if (value == null || isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
}

/**
 * Format a timestamp to a locale-aware time string.
 * @param {string|Date|number} ts
 * @returns {string}
 */
export function formatTimestamp(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

/**
 * Format a timestamp to show just the hour (e.g. "14:00").
 * @param {string|Date|number} ts
 * @returns {string}
 */
export function formatHour(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  return `${h}:${m}`;
}

/**
 * Format a full date-time string.
 * @param {string|Date|number} ts
 * @returns {string}
 */
export function formatDateTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

/**
 * Get the percentage position of an AQI value within the 0-500 scale.
 * @param {number} aqi
 * @returns {number} 0–100
 */
export function getAQIPercent(aqi) {
  return Math.min(100, Math.max(0, (Math.round(aqi) / 500) * 100));
}

export { AQI_CATEGORIES };
