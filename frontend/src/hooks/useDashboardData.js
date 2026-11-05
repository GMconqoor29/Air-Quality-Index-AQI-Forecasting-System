import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchAlertHistory,
  fetchCurrentReading,
  fetchHistoricalData,
  fetchModelComparison,
  syncTelemetryBackend,
} from '../api/aqiApi';

const REFRESH_INTERVAL_MS = 60_000;
const DEFAULT_HOURS = 168; // 7 days, enough for every chart range

/**
 * Central data hook for the dashboard. Loads the current reading, historical
 * series, model comparison and alert history in parallel, and optionally polls
 * on a fixed interval. The forecast is handled by `ForecastCard` on its own so
 * its horizon/model controls can refetch just that slice.
 */
export function useDashboardData(hours = DEFAULT_HOURS) {
  const [current, setCurrent] = useState(null);
  const [historical, setHistorical] = useState([]);
  const [modelComparison, setModelComparison] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const inFlight = useRef(false);

  const load = useCallback(
    async (isRefresh = false) => {
      if (inFlight.current) return;
      inFlight.current = true;
      if (isRefresh) setRefreshing(true);
      try {
        const [current, historical, modelComparison, alerts] = await Promise.all([
          fetchCurrentReading(),
          fetchHistoricalData(hours),
          fetchModelComparison(),
          fetchAlertHistory(),
        ]);
        setCurrent(current);
        setHistorical(historical);
        setModelComparison(modelComparison);
        setAlerts(alerts);
        setError(null);
        setLastUpdated(new Date());
      } catch (err) {
        setError(err?.message || 'Unable to load data.');
      } finally {
        setLoading(false);
        setRefreshing(false);
        inFlight.current = false;
      }
    },
    [hours],
  );

  useEffect(() => {
    load(false);
  }, [load]);

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(() => load(true), REFRESH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [autoRefresh, load]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    await syncTelemetryBackend();
    return load(true);
  }, [load]);

  return {
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
  };
}
