import { useState } from 'react';
import {
  FiActivity,
  FiAlertTriangle,
  FiBarChart2,
  FiCloud,
  FiCpu,
  FiInfo,
  FiMapPin,
  FiMenu,
  FiRefreshCw,
  FiTrendingUp,
} from 'react-icons/fi';
import { useDashboardData } from './hooks/useDashboardData';
import { useTheme } from './hooks/useTheme';
import Navbar from './components/Navbar/Navbar';
import Card from './components/Card/Card';
import AQIGauge from './components/AQIGauge/AQIGauge';
import SensorReadings from './components/SensorReadings/SensorReadings';
import SensorChart from './components/SensorChart/SensorChart';
import ModelComparison from './components/ModelComparison/ModelComparison';
import AlertPanel from './components/AlertPanel/AlertPanel';
import { AQI_CATEGORIES, formatDateTime, getAQICategory } from './utils/aqiHelpers';
import './App.css';

const PAGE_META = {
  dashboard: {
    code: 'VIEW // 01',
    title: 'Atmospheric Telemetry & Horizon Forecast',
    subtitle: 'Real-time particulate index, 5-channel IoT sensor array, and neural/GBDT projections for Hanoi',
  },
  models: {
    code: 'VIEW // 02',
    title: 'Architecture Benchmark & Regime Trade-offs',
    subtitle: 'Holdout evaluation across 10 forecasting models on aggregate error (RMSE/MAE) vs. peak capture (AQI ≥ 200)',
  },
  alerts: {
    code: 'VIEW // 03',
    title: 'High-AQI Anomaly & Incident Log',
    subtitle: 'Chronological record of observed hazardous spikes and projected peak-regime threshold crossings',
  },
  about: {
    code: 'VIEW // 04',
    title: 'System Dossier & Forecasting Methodology',
    subtitle: 'Sensor tensor specification, asymmetric loss formulation, and US EPA breakpoint calibration',
  },
};

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const [activePage, setActivePage] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const {
    current,
    historical,
    modelComparison,
    alerts,
    loading,
    refreshing,
    error,
    lastUpdated,
    autoRefresh,
    setAutoRefresh,
    refresh,
  } = useDashboardData(168);

  const meta = PAGE_META[activePage];
  const currentCategory = getAQICategory(current?.aqi);

  return (
    <div
      className="app"
      style={{ '--ambient-aqi-color': currentCategory.color }}
    >
      <div className="app-content">
        <Navbar
          activePage={activePage}
          onNavigate={setActivePage}
          theme={theme}
          onToggleTheme={toggleTheme}
          autoRefresh={autoRefresh}
          onToggleAutoRefresh={() => setAutoRefresh((v) => !v)}
          refreshing={refreshing}
          lastUpdated={lastUpdated}
          currentAqi={current?.aqi}
          alertCount={alerts.length}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />

        <main className="main-content">
          {/* Top Atmospheric Telemetry Ribbon */}
          <div className="telemetry-ribbon mono" aria-label="Station metadata">
            <div className="telemetry-ribbon-group">
              <span className="telemetry-pill">
                <FiMapPin aria-hidden="true" />
                HANOI BASIN · 21.0285°N 105.8542°E
              </span>
              <span className="telemetry-sep">/</span>
              <span className="telemetry-item">TENSOR: 48H × 7 CHANNELS</span>
              <span className="telemetry-sep">/</span>
              <span className="telemetry-item">HORIZON: T+1H</span>
            </div>
            <div className="telemetry-ribbon-group">
              {lastUpdated && (
                <span className="telemetry-item">
                  LAST SYNC: {formatDateTime(lastUpdated)}
                </span>
              )}
              <span className={`telemetry-status ${autoRefresh ? 'is-live' : ''}`}>
                <span className="telemetry-status-dot" />
                {autoRefresh ? 'STREAM ACTIVE' : 'STREAM PAUSED'}
              </span>
            </div>
          </div>

          <header className="page-header">
            <div className="page-header-title">
              <div className="page-kicker-row">
                <button
                  type="button"
                  className="btn btn-ghost btn-icon menu-btn"
                  onClick={() => setSidebarOpen(true)}
                  aria-label="Open navigation menu"
                >
                  <FiMenu aria-hidden="true" />
                </button>
                <span className="page-kicker mono">{meta.code}</span>
              </div>
              <h1 className="page-title">{meta.title}</h1>
              <p className="page-subtitle">{meta.subtitle}</p>
            </div>
            <div className="page-actions">
              <button
                type="button"
                className="btn mono"
                onClick={refresh}
                disabled={refreshing}
              >
                <FiRefreshCw className={refreshing ? 'spinner' : ''} aria-hidden="true" />
                {refreshing ? 'SYNCING…' : 'SYNC TELEMETRY'}
              </button>
            </div>
          </header>

          {error && (
            <div className="banner banner-error" role="alert">
              Could not load dashboard data: {error}. The most recently loaded readings are retained where available.
            </div>
          )}

          <div className="page-body fade-in" key={activePage}>
            {activePage === 'dashboard' && (
              <DashboardPage
                current={current}
                historical={historical}
                loading={loading}
                error={error}
                onViewModels={() => setActivePage('models')}
              />
            )}

            {activePage === 'models' && (
              <Card
                code="MOD-BENCH"
                title="Holdout Architecture Leaderboard"
                subtitle="Chronological evaluation on ~30,300 hourly Hanoi observations (2022–2025)"
                icon={FiBarChart2}
                className="full-width-section"
              >
                <ModelComparison data={modelComparison} loading={loading} error={error} />
              </Card>
            )}

            {activePage === 'alerts' && (
              <Card
                code="MOD-ALERTS"
                title="Atmospheric Anomaly & Spike Log"
                subtitle="Events triggered when observed or projected AQI crosses US EPA critical thresholds"
                icon={FiAlertTriangle}
                className="full-width-section"
              >
                <AlertPanel alerts={alerts} loading={loading} error={error} />
              </Card>
            )}

            {activePage === 'about' && <AboutPage />}
          </div>
        </main>
      </div>
    </div>
  );
}

function DashboardPage({
  current,
  historical,
  loading,
  error,
  onViewModels,
}) {
  return (
    <div className="dashboard-grid">
      <Card
        code="MOD-01"
        title="Current Atmospheric Index"
        subtitle="Instantaneous US-EPA AQI & 24-hour drift"
        icon={FiActivity}
      >
        <AQIGauge value={current?.aqi} current={current} historical={historical} />
      </Card>

      <Card
        code="MOD-02"
        title="IoT Sensor Array Telemetry"
        subtitle="Particulate matter & meteorological boundary-layer channels (24h sparklines)"
        icon={FiCloud}
      >
        <SensorReadings current={current} historical={historical} />
      </Card>

      <Card
        code="MOD-03"
        title="AI Forecast (t+1h) & Historical Verification"
        subtitle="Synced model selector: 1-hour ahead projection, 90% CI & rolling Predicted vs. OpenAQ waveform"
        icon={FiTrendingUp}
        className="col-span-2"
        action={
          <button type="button" className="card-link-btn mono" onClick={onViewModels}>
            COMPARE 10 MODELS →
          </button>
        }
      >
        <SensorChart
          data={historical}
          currentAqi={current?.aqi}
          loading={loading}
          error={error}
        />
      </Card>
    </div>
  );
}

function AboutPage() {
  return (
    <div className="about">
      <Card
        code="SYS-01"
        title="Forecasting Pipeline Architecture"
        subtitle="End-to-end hourly inference from 48-hour multivariate IoT windows"
        icon={FiCpu}
        className="full-width-section"
      >
        <p>
          This observatory forecasts Hanoi’s hourly Air Quality Index (AQI) at the{' '}
          <strong>t+1h</strong> horizon from a sliding{' '}
          <strong>48-hour × 7-channel</strong> window of environmental sensor readings — fine
          particulate matter (PM₂.₅), coarse particulate matter (PM₁₀), ambient temperature,
          relative humidity, barometric pressure, cyclical hour-of-day, and month.
        </p>
        <p>
          Standard MSE-trained regressors systematically under-predict hazardous pollution episodes
          (<strong>AQI ≥ 200</strong>) where public-health consequences are most severe. Our system
          benchmarks 10 architectures against a persistence baseline, introducing Asymmetric MSE
          (AMSE) weighting, learned soft-gating, and a Two-Stage Peak Specialist.
        </p>

        <div className="pipeline-diagram">
          <div className="pipeline-stage">
            <span className="pipeline-step mono">STAGE 01 // INPUT TENSOR</span>
            <div className="pipeline-title">48h × 7 Sensor Window</div>
            <div className="pipeline-desc mono">PM2.5 · PM10 · T · RH · P · Hour · Month</div>
          </div>
          <div className="pipeline-stage-arrow mono">→</div>
          <div className="pipeline-stage">
            <span className="pipeline-step mono">STAGE 02 // REPRESENTATION</span>
            <div className="pipeline-title">336-Lag Flat vs. 1D-Conv / RNN</div>
            <div className="pipeline-desc mono">Chronological leakage-free split</div>
          </div>
          <div className="pipeline-stage-arrow mono">→</div>
          <div className="pipeline-stage">
            <span className="pipeline-step mono">STAGE 03 // SPECIALIST ROUTING</span>
            <div className="pipeline-title">Soft-Gate & Two-Stage Peak</div>
            <div className="pipeline-desc mono">AMSE penalty on AQI ≥ 200 regime</div>
          </div>
          <div className="pipeline-stage-arrow mono">→</div>
          <div className="pipeline-stage">
            <span className="pipeline-step mono">STAGE 04 // OUTPUT</span>
            <div className="pipeline-title">t+1h AQI + 90% CI</div>
            <div className="pipeline-desc mono">Calibrated EPA category & alerts</div>
          </div>
        </div>
      </Card>

      <Card
        code="SYS-02"
        title="US EPA Breakpoint & Health Advisory Matrix"
        subtitle="Standardized 6-tier categorization used across gauges, waveforms, and alerts"
        icon={FiActivity}
        className="full-width-section"
      >
        <div className="aqi-legend">
          {AQI_CATEGORIES.map((cat) => (
            <div
              className="aqi-legend-item"
              key={cat.level}
              style={{ borderLeftColor: cat.color }}
            >
              <div className="aqi-legend-top">
                <span className="aqi-legend-code mono" style={{ color: cat.color }}>
                  {cat.code}
                </span>
                <span className="aqi-legend-range mono">{cat.range} AQI</span>
              </div>
              <div className="aqi-legend-label">{cat.label}</div>
              <p className="aqi-legend-desc">{cat.description}</p>
              <p className="aqi-legend-advice">{cat.advice}</p>
            </div>
          ))}
        </div>
      </Card>

      <div className="dashboard-grid">
        <Card
          code="SYS-03"
          title="Evaluated Model Families"
          subtitle="From naive persistence to attention hybrids"
          icon={FiBarChart2}
        >
          <ul className="about-list">
            <li>
              <strong>Recurrent Baselines:</strong> Base LSTM, Stacked 2-layer LSTM, and single-layer
              GRU implemented in PyTorch.
            </li>
            <li>
              <strong>Gradient Boosted Trees:</strong> Untuned and Optuna-tuned XGBoost operating on
              flattened 48×7 (336-feature) lag vectors.
            </li>
            <li>
              <strong>Conv-Attention Hybrids:</strong> 1D-CNN feature extractors coupled with GRU/LSTM
              and temporal attention for short-range spike motifs.
            </li>
            <li>
              <strong>Asymmetric & Two-Stage Specialists:</strong> AQI-weighted AMSE soft-gated
              ensemble and a two-stage peak classifier + specialist regressor (90.4% peak recall).
            </li>
          </ul>
        </Card>

        <Card
          code="SYS-04"
          title="Dataset & Telemetry Provenance"
          subtitle="Hanoi urban basin corpus (2022–2025)"
          icon={FiCloud}
        >
          <p>
            Model training and evaluation span Hanoi from <strong>2022-01-13</strong> to{' '}
            <strong>2025-06-30</strong> on an hourly grid (~30,300 observations), capturing winter
            thermal inversions and seasonal monsoon transitions.
          </p>
          <p>
            This web interface operates on a deterministic diurnal telemetry simulator (
            <code>frontend/src/api/aqiApi.js</code>) that preserves realistic rush-hour peaks,
            overnight decay, and holdout benchmark metrics while remaining ready for drop-in Flask
            REST API integration.
          </p>
        </Card>
      </div>
    </div>
  );
}
