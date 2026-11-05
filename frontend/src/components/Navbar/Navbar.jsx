import {
  FiActivity,
  FiAlertTriangle,
  FiBarChart2,
  FiClock,
  FiCompass,
  FiCpu,
  FiInfo,
  FiMoon,
  FiRadio,
  FiRefreshCw,
  FiSun,
} from 'react-icons/fi';
import { formatAQI, formatDateTime, getAQICategory } from '../../utils/aqiHelpers';

const NAV_ITEMS = [
  { id: 'dashboard', index: '01', label: 'Telemetry Overview', sub: 'Live & t+1h forecast', icon: FiCompass },
  { id: 'models', index: '02', label: 'Model Benchmark', sub: '10 architectures compared', icon: FiBarChart2 },
  { id: 'alerts', index: '03', label: 'Anomaly Log', sub: 'Peak regime (AQI ≥ 200)', icon: FiAlertTriangle },
  { id: 'about', index: '04', label: 'System Dossier', sub: 'Methodology & EPA bands', icon: FiInfo },
];

/**
 * Sidebar navigation — styled as an Atmospheric Observatory Command Rail.
 */
export default function Navbar({
  activePage,
  onNavigate,
  theme,
  onToggleTheme,
  autoRefresh,
  onToggleAutoRefresh,
  refreshing,
  lastUpdated,
  currentAqi,
  alertCount = 0,
  open,
  onClose,
}) {
  const category = getAQICategory(currentAqi);

  return (
    <>
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="sidebar-logo">
          <div className="sidebar-brand-mark" aria-hidden="true">
            <FiActivity className="sidebar-logo-icon" />
            <span className="sidebar-brand-ping" />
          </div>
          <div className="sidebar-brand-copy">
            <div className="sidebar-logo-kicker mono">HN-ATMOS // NODE 01</div>
            <div className="sidebar-logo-text">Hanoi AQI</div>
            <div className="sidebar-logo-sub">Atmospheric Intelligence</div>
          </div>
        </div>

        <div className="sidebar-station-card">
          <div className="sidebar-station-row">
            <span className="sidebar-station-label mono">COORDS</span>
            <span className="sidebar-station-val mono">21.0285°N, 105.8542°E</span>
          </div>
          <div className="sidebar-station-row">
            <span className="sidebar-station-label mono">WINDOW</span>
            <span className="sidebar-station-val mono">48H × 7 CHANNELS</span>
          </div>
          {currentAqi != null && (
            <div className="sidebar-station-live">
              <div className="sidebar-station-live-left">
                <span
                  className="sidebar-live-dot"
                  style={{ background: category.color, boxShadow: `0 0 10px ${category.color}` }}
                />
                <span className="mono">{category.shortLabel || category.label}</span>
              </div>
              <span className="sidebar-station-aqi mono" style={{ color: category.color }}>
                AQI {formatAQI(currentAqi)}
              </span>
            </div>
          )}
        </div>

        <nav className="sidebar-nav" aria-label="Primary">
          <div className="sidebar-section-title mono">OBSERVATORY MODULES</div>
          {NAV_ITEMS.map(({ id, index, label, sub, icon: Icon }) => {
            const isActive = activePage === id;
            return (
              <button
                key={id}
                type="button"
                className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
                aria-current={isActive ? 'page' : undefined}
                onClick={() => {
                  onNavigate(id);
                  onClose();
                }}
              >
                <span className="sidebar-nav-index mono">{index}</span>
                <Icon className="sidebar-nav-icon" aria-hidden="true" />
                <span className="sidebar-nav-text">
                  <span className="sidebar-nav-label">{label}</span>
                  <span className="sidebar-nav-sub">{sub}</span>
                </span>
                {id === 'alerts' && alertCount > 0 && (
                  <span className="sidebar-nav-badge mono">{alertCount}</span>
                )}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <FiRadio className="sidebar-status-icon" aria-hidden="true" />
            <div className="sidebar-status-copy">
              <span className="sidebar-status-title">Simulated Telemetry</span>
              <span className="sidebar-status-sub mono">Diurnal PRNG · 168h buffer</span>
            </div>
          </div>

          <div className="sidebar-footer-row">
            <button
              type="button"
              className="btn btn-ghost"
              onClick={onToggleAutoRefresh}
              title={autoRefresh ? 'Pause 60s polling' : 'Resume 60s polling'}
            >
              <FiRefreshCw className={refreshing ? 'spinner' : ''} aria-hidden="true" />
              <span>{autoRefresh ? 'Polling: ON' : 'Polling: OFF'}</span>
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-icon"
              onClick={onToggleTheme}
              aria-label={theme === 'dark' ? 'Switch to archival light theme' : 'Switch to nocturnal dark theme'}
              title={theme === 'dark' ? 'Switch to archival light theme' : 'Switch to nocturnal dark theme'}
            >
              {theme === 'dark' ? <FiSun aria-hidden="true" /> : <FiMoon aria-hidden="true" />}
            </button>
          </div>

          {lastUpdated && (
            <div className="sidebar-updated mono">
              <FiClock aria-hidden="true" />
              <span>SYNC // {formatDateTime(lastUpdated)}</span>
            </div>
          )}
        </div>
      </aside>

      <div className={`sidebar-overlay ${open ? 'open' : ''}`} onClick={onClose} aria-hidden="true" />
    </>
  );
}
