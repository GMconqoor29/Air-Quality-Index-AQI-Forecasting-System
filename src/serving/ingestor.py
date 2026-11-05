"""Real-time SQLite store, OpenAQ v3 (Location 4946813 - ĐHBK) client, and ESP32 ingestor."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
import sqlite3
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

from src.serving.aqi_calculator import compute_aqi_from_pm

VN_TZ = timezone(timedelta(hours=7))


def load_env_file(env_path: str = ".env") -> dict[str, str]:
    """Parse key=value pairs from `.env` into os.environ and return them."""
    loaded = {}
    if not os.path.exists(env_path):
        return loaded
    with open(env_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            os.environ.setdefault(k, v)
            loaded[k] = v
    return loaded


class RealtimeDataStore:
    """Manages SQLite persistence, OpenAQ v3 synchronization, and ESP32 telemetry."""

    def __init__(self, db_path: str | None = None):
        load_env_file()
        self.db_path = db_path or os.getenv("DB_PATH", "results/realtime_aqi.db")
        self.openaq_api_key = os.getenv("OPENAQ_API_KEY", "").strip()
        self.openaq_location_id = int(os.getenv("OPENAQ_LOCATION_ID", "4946813"))
        self.station_lat = float(os.getenv("STATION_LAT", "21.0056"))
        self.station_lon = float(os.getenv("STATION_LON", "105.8433"))
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    timestamp TEXT PRIMARY KEY,
                    aqi INTEGER NOT NULL,
                    pm25 REAL NOT NULL,
                    pm10 REAL NOT NULL,
                    temperature REAL NOT NULL,
                    relative_humidity REAL NOT NULL,
                    pressure REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT 'seed',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issued_at TEXT NOT NULL,
                    target_timestamp TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    model_used TEXT NOT NULL,
                    predicted_aqi INTEGER NOT NULL,
                    ci_low INTEGER NOT NULL,
                    ci_high INTEGER NOT NULL,
                    peak_prob REAL
                );
                """
            )

    def ensure_warm_start(self, hours: int = 168) -> int:
        """Ensure at least `hours` contiguous hourly observations exist up to the current hour.

        Uses the tail of Hanoi historical observations (`Dataset/archive/2025.csv`)
        mapped onto the trailing `hours` window as a warm-start baseline, preserving
        any existing `openaq` or `esp32_sensor` rows.
        """
        now_hour = datetime.now(VN_TZ).replace(minute=0, second=0, microsecond=0)
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM observations").fetchone()
            existing_count = int(row["cnt"]) if row else 0

        # Check if the latest hour is already present and we have enough history
        latest = self.get_current_reading()
        if existing_count >= hours and latest is not None:
            latest_dt = pd.to_datetime(latest["timestamp"])
            if latest_dt.tzinfo is None:
                latest_dt = latest_dt.tz_localize(VN_TZ)
            if (now_hour - latest_dt).total_seconds() <= 3600:
                return existing_count

        csv_path = os.path.join("Dataset", "archive", "2025.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["Local Time"] = pd.to_datetime(df["Local Time"])
            df = df.sort_values("Local Time").drop_duplicates("Local Time").tail(hours + 24)
            df = df.ffill().bfill().tail(hours).reset_index(drop=True)
        else:
            df = None

        now_iso = datetime.now(VN_TZ).isoformat()
        inserted = 0
        with self._connect() as conn:
            for idx in range(hours):
                ts = now_hour - timedelta(hours=(hours - 1 - idx))
                ts_iso = ts.isoformat()

                # Do not overwrite live OpenAQ or ESP32 readings
                cur = conn.execute(
                    "SELECT source FROM observations WHERE timestamp = ?", (ts_iso,)
                ).fetchone()
                if cur and cur["source"] in ("openaq", "esp32_sensor"):
                    continue

                if df is not None and idx < len(df):
                    r = df.iloc[idx]
                    pm25 = float(r.get("PM25", 35.0))
                    pm10 = float(r.get("PM10", max(pm25 * 1.35, 45.0)))
                    temp = float(r.get("Temperature", 26.0))
                    rh = float(r.get("Relative Humidity", 75.0))
                    pres = float(r.get("Pressure", 1009.0))
                else:
                    pm25, pm10, temp, rh, pres = 35.0, 50.0, 26.0, 75.0, 1009.0

                aqi = compute_aqi_from_pm(pm25, pm10)
                if aqi is None:
                    aqi = 100

                conn.execute(
                    """
                    INSERT OR REPLACE INTO observations
                    (timestamp, aqi, pm25, pm10, temperature, relative_humidity, pressure, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts_iso,
                        int(aqi),
                        round(pm25, 1),
                        round(pm10, 1),
                        round(temp, 1),
                        round(rh, 1),
                        round(pres, 1),
                        "warm_start",
                        now_iso,
                    ),
                )
                inserted += 1
        return inserted

    def _http_get_json(self, url: str, headers: dict[str, str] | None = None, timeout: float = 10.0) -> dict:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def sync_openaq_and_weather(self, hours: int = 72) -> dict:
        """Fetch hourly PM2.5/PM10 from OpenAQ v3 (Location 4946813 - ĐHBK) + meteo from Open-Meteo."""
        load_env_file()
        self.openaq_api_key = os.getenv("OPENAQ_API_KEY", self.openaq_api_key).strip()
        summary = {
            "openaq_location_id": self.openaq_location_id,
            "openaq_station": "Số 1 đường Giải Phóng - phường Bạch Mai - ĐHBK",
            "openaq_records_fetched": 0,
            "weather_records_fetched": 0,
            "rows_updated": 0,
            "status": "ok",
            "errors": [],
        }

        # 1. Fetch hourly weather (Temperature, Relative Humidity, Pressure) from Open-Meteo
        weather_by_hour: dict[str, dict] = {}
        try:
            past_days = min(92, max(7, int(hours // 24) + 2))
            meteo_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={self.station_lat}&longitude={self.station_lon}"
                f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
                f"&past_days={past_days}&forecast_days=1&timezone=Asia%2FBangkok"
            )
            meteo_data = self._http_get_json(meteo_url, timeout=8.0)
            hourly = meteo_data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            rhs = hourly.get("relative_humidity_2m", [])
            press = hourly.get("surface_pressure", [])
            for t_str, temp, rh, pr in zip(times, temps, rhs, press):
                dt = pd.to_datetime(t_str).tz_localize(VN_TZ)
                weather_by_hour[dt.replace(minute=0, second=0, microsecond=0).isoformat()] = {
                    "temperature": float(temp) if temp is not None else None,
                    "relative_humidity": float(rh) if rh is not None else None,
                    "pressure": float(pr) if pr is not None else None,
                }
            summary["weather_records_fetched"] = len(weather_by_hour)
        except Exception as exc:
            summary["errors"].append(f"Open-Meteo: {exc}")

        # 2. Fetch PM2.5 & PM10 from OpenAQ v3 for Location 4946813 (Số 1 Giải Phóng - ĐHBK)
        pm_by_hour: dict[str, dict[str, float]] = {}
        if not self.openaq_api_key or self.openaq_api_key == "PASTE_YOUR_OPENAQ_API_KEY_HERE":
            summary["errors"].append("OPENAQ_API_KEY not configured in .env")
        else:
            headers = {
                "X-API-Key": self.openaq_api_key,
                "Accept": "application/json",
                "User-Agent": "Hanoi-AQI-Forecasting-System/1.0",
            }
            try:
                loc_url = f"https://api.openaq.org/v3/locations/{self.openaq_location_id}"
                loc_payload = self._http_get_json(loc_url, headers=headers, timeout=10.0)
                results = loc_payload.get("results", [])
                sensors = results[0].get("sensors", []) if results else []
                sensor_map = {}
                for s in sensors:
                    param_name = (s.get("parameter", {}) or {}).get("name", "").lower()
                    if param_name in ("pm25", "pm2.5", "pm10"):
                        norm_name = "pm25" if "25" in param_name or "2.5" in param_name else "pm10"
                        sensor_map[norm_name] = s.get("id")

                date_from = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
                page_limit = min(1000, max(100, int(hours) + 24))
                past_days = min(92, max(7, int(hours // 24) + 2))
                for param_key, sensor_id in sensor_map.items():
                    if not sensor_id:
                        continue
                    hrs_url = (
                        f"https://api.openaq.org/v3/sensors/{sensor_id}/hours?"
                        f"datetime_from={urllib.parse.quote(date_from)}&limit={page_limit}"
                    )
                    hrs_payload = self._http_get_json(hrs_url, headers=headers, timeout=10.0)
                    for item in hrs_payload.get("results", []):
                        val = item.get("value")
                        period = item.get("period", {}) or {}
                        dt_obj = period.get("datetimeTo") or period.get("datetimeFrom") or {}
                        utc_str = dt_obj.get("utc") or dt_obj.get("local")
                        if val is None or not utc_str:
                            continue
                        dt = pd.to_datetime(utc_str)
                        if dt.tzinfo is None:
                            dt = dt.tz_localize(timezone.utc)
                        dt_vn = dt.astimezone(VN_TZ).replace(minute=0, second=0, microsecond=0)
                        iso_key = dt_vn.isoformat()
                        pm_by_hour.setdefault(iso_key, {})[param_key] = float(val)

                # 2b. Also fetch /latest from OpenAQ v3 to capture the newest reading if the current hour is still open
                try:
                    latest_url = f"https://api.openaq.org/v3/locations/{self.openaq_location_id}/latest"
                    latest_payload = self._http_get_json(latest_url, headers=headers, timeout=10.0)
                    sensor_id_to_param = {v: k for k, v in sensor_map.items() if v}
                    for item in latest_payload.get("results", []):
                        sid = item.get("sensorsId")
                        val = item.get("value")
                        dt_str = (item.get("datetime") or {}).get("utc") or (item.get("datetime") or {}).get("local")
                        param_key = sensor_id_to_param.get(sid)
                        if param_key and val is not None and dt_str:
                            dt = pd.to_datetime(dt_str)
                            if dt.tzinfo is None:
                                dt = dt.tz_localize(timezone.utc)
                            dt_vn = dt.astimezone(VN_TZ).replace(minute=0, second=0, microsecond=0)
                            iso_key = dt_vn.isoformat()
                            pm_by_hour.setdefault(iso_key, {}).setdefault(param_key, float(val))
                except Exception:
                    pass

                summary["openaq_records_fetched"] = len(pm_by_hour)
            except Exception as exc:
                summary["errors"].append(f"OpenAQ v3: {exc}")

        # 3. Forward-fill 1-hour gaps in OpenAQ PM2.5/PM10 (including the current hour if OpenAQ lags by 1h)
        now_hour_iso = datetime.now(VN_TZ).replace(minute=0, second=0, microsecond=0).isoformat()
        all_hours = sorted(
            h for h in (set(pm_by_hour.keys()) | set(weather_by_hour.keys()))
            if h <= now_hour_iso
        )
        last_valid_pm: dict[str, float] = {}
        last_valid_ts: pd.Timestamp | None = None
        for ts_iso in all_hours:
            cur_dt = pd.to_datetime(ts_iso)
            if ts_iso in pm_by_hour and "pm25" in pm_by_hour[ts_iso]:
                last_valid_pm = dict(pm_by_hour[ts_iso])
                last_valid_ts = cur_dt
            elif last_valid_ts is not None and (cur_dt - last_valid_ts).total_seconds() <= 7200:
                # Forward-fill up to 2 hours from the latest real OpenAQ reading so current hour isn't stale CSV
                pm_by_hour[ts_iso] = dict(last_valid_pm)

        now_iso = datetime.now(VN_TZ).isoformat()
        updated = 0
        with self._connect() as conn:
            for ts_iso in all_hours:
                pm_info = pm_by_hour.get(ts_iso, {})
                w_info = weather_by_hour.get(ts_iso, {})
                if not pm_info and not w_info:
                    continue

                existing = conn.execute(
                    "SELECT * FROM observations WHERE timestamp = ?", (ts_iso,)
                ).fetchone()
                # Preserve direct ESP32 sensor readings unless missing
                if existing and existing["source"] == "esp32_sensor":
                    continue

                pm25 = pm_info.get("pm25", existing["pm25"] if existing else None)
                if pm25 is None:
                    continue
                pm10 = pm_info.get("pm10", existing["pm10"] if existing else round(pm25 * 1.38, 1))
                temp = (
                    w_info["temperature"]
                    if w_info.get("temperature") is not None
                    else (existing["temperature"] if existing else 26.0)
                )
                rh = (
                    w_info["relative_humidity"]
                    if w_info.get("relative_humidity") is not None
                    else (existing["relative_humidity"] if existing else 75.0)
                )
                pres = (
                    w_info["pressure"]
                    if w_info.get("pressure") is not None
                    else (existing["pressure"] if existing else 1010.0)
                )

                aqi = compute_aqi_from_pm(pm25, pm10)
                if aqi is None:
                    continue

                source_tag = "openaq" if "pm25" in pm_info else (existing["source"] if existing else "warm_start")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO observations
                    (timestamp, aqi, pm25, pm10, temperature, relative_humidity, pressure, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts_iso,
                        int(aqi),
                        round(float(pm25), 1),
                        round(float(pm10), 1),
                        round(float(temp), 1),
                        round(float(rh), 1),
                        round(float(pres), 1),
                        source_tag,
                        now_iso,
                    ),
                )
                updated += 1

        summary["rows_updated"] = updated
        if summary["errors"] and updated == 0:
            summary["status"] = "degraded"
        self.export_collected_csv()
        return summary

    def export_collected_csv(
        self,
        master_csv_path: str = "Dataset/openaq_dhbk_collected.csv",
        daily_dir: str = "Dataset/openaq_daily",
    ) -> dict:
        """Persist collected OpenAQ & ESP32 observations to CSV files (per FAIR2023 / Thesis Slide 24).

        Applies the 3 EDA cleaning steps on Slide 24:
          1. Reform and equidistant timestamps (hourly grid alignment)
          2. Remove outlier data (AQI > 400 or negative concentrations)
          3. Handle missing data (causal forward-fill up to 1 hour)
        Saves both:
          - Master dataset CSV (`Dataset/openaq_dhbk_collected.csv`) matching training schema
          - Daily CSV snapshot (`Dataset/openaq_daily/YYYY-MM-DD.csv`) like ThingsBoardController.py
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, aqi, pm25, pm10, temperature, relative_humidity, pressure, source
                FROM observations
                WHERE source IN ('openaq', 'esp32_sensor')
                ORDER BY timestamp ASC
                """
            ).fetchall()

        if not rows:
            return {"saved_rows": 0, "master_csv": master_csv_path}

        records = []
        for r in rows:
            dt = pd.to_datetime(r["timestamp"])
            if dt.tzinfo is not None:
                dt_local = dt.tz_convert(VN_TZ).tz_localize(None)
            else:
                dt_local = dt
            records.append(
                {
                    "Local Time": dt_local.strftime("%Y-%m-%d %H:%M:%S"),
                    "AQI": int(r["aqi"]),
                    "PM25": float(r["pm25"]),
                    "PM10": float(r["pm10"]),
                    "Temperature": float(r["temperature"]),
                    "Relative Humidity": float(r["relative_humidity"]),
                    "Pressure": float(r["pressure"]),
                    "source": r["source"],
                }
            )

        df = pd.DataFrame(records)
        df["Local Time"] = pd.to_datetime(df["Local Time"])
        df = df.sort_values("Local Time").drop_duplicates("Local Time")

        # Step 1: Equidistant hourly timestamps
        df = df.set_index("Local Time").asfreq("h")

        # Step 2: Remove outliers
        df.loc[df["AQI"] > 400, "AQI"] = np.nan
        df.loc[(df["PM25"] < 0) | (df["PM25"] > 1000), "PM25"] = np.nan
        df.loc[(df["PM10"] < 0) | (df["PM10"] > 1500), "PM10"] = np.nan

        # Step 3: Handle missing data (forward fill <= 1h)
        num_cols = ["AQI", "PM25", "PM10", "Temperature", "Relative Humidity", "Pressure"]
        df[num_cols] = df[num_cols].ffill(limit=1)
        df["source"] = df["source"].ffill(limit=1).fillna("openaq")
        df = df.dropna(subset=["PM25", "AQI"]).reset_index()
        df["AQI"] = df["AQI"].round().astype(int)
        df["Local Time"] = df["Local Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        os.makedirs(os.path.dirname(master_csv_path) or ".", exist_ok=True)
        df.to_csv(master_csv_path, index=False)

        # Save latest day CSV snapshot (`Dataset/openaq_daily/YYYY-MM-DD.csv`)
        os.makedirs(daily_dir, exist_ok=True)
        latest_day = df["Local Time"].iloc[-1][:10]
        df_day = df[df["Local Time"].str.startswith(latest_day)]
        daily_path = os.path.join(daily_dir, f"{latest_day}.csv")
        df_day.to_csv(daily_path, index=False)

        return {
            "saved_rows": len(df),
            "master_csv": master_csv_path,
            "daily_csv": daily_path,
        }

    def ingest_esp32_telemetry(self, payload: dict) -> dict:
        """Ingest a telemetry packet from the ESP32 (PMS7003 + BME280) node."""
        now_vn = datetime.now(VN_TZ)
        raw_ts = payload.get("timestamp") or payload.get("ts")
        if raw_ts:
            dt = pd.to_datetime(raw_ts)
            if dt.tzinfo is None:
                dt = dt.tz_localize(VN_TZ)
            else:
                dt = dt.astimezone(VN_TZ)
        else:
            dt = now_vn

        hour_ts = dt.replace(minute=0, second=0, microsecond=0).isoformat()
        pm25 = float(payload["pm25"])
        pm10 = float(payload.get("pm10", round(pm25 * 1.35, 1)))
        temperature = float(payload.get("temperature", payload.get("temp", 26.0)))
        rh = float(payload.get("relativeHumidity", payload.get("relative_humidity", payload.get("rh", 75.0))))
        pressure = float(payload.get("pressure", 1010.0))

        aqi = compute_aqi_from_pm(pm25, pm10)
        if aqi is None:
            raise ValueError("Invalid PM2.5 / PM10 values in telemetry payload.")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO observations
                (timestamp, aqi, pm25, pm10, temperature, relative_humidity, pressure, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'esp32_sensor', ?)
                """,
                (
                    hour_ts,
                    int(aqi),
                    round(pm25, 1),
                    round(pm10, 1),
                    round(temperature, 1),
                    round(rh, 1),
                    round(pressure, 1),
                    now_vn.isoformat(),
                ),
            )

        self.export_collected_csv()
        return {
            "timestamp": hour_ts,
            "aqi": int(aqi),
            "pm25": round(pm25, 1),
            "pm10": round(pm10, 1),
            "temperature": round(temperature, 1),
            "relativeHumidity": round(rh, 1),
            "pressure": round(pressure, 1),
            "source": "esp32_sensor",
        }

    def get_historical_readings(self, hours: int = 168) -> list[dict]:
        """Return the last `hours` observations formatted for the React frontend."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, aqi, pm25, pm10, temperature, relative_humidity, pressure, source
                FROM observations
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(hours),),
            ).fetchall()

        out = []
        for r in reversed(rows):
            out.append(
                {
                    "timestamp": r["timestamp"],
                    "aqi": int(r["aqi"]),
                    "pm25": float(r["pm25"]),
                    "pm10": float(r["pm10"]),
                    "temperature": float(r["temperature"]),
                    "relativeHumidity": float(r["relative_humidity"]),
                    "pressure": float(r["pressure"]),
                    "source": r["source"],
                }
            )
        return out

    def get_current_reading(self) -> dict | None:
        rows = self.get_historical_readings(hours=1)
        return rows[-1] if rows else None

    def record_forecast(self, forecast: dict) -> None:
        now_iso = datetime.now(VN_TZ).isoformat()
        target_ts = (
            datetime.now(VN_TZ).replace(minute=0, second=0, microsecond=0)
            + timedelta(hours=int(forecast["horizon"]))
        ).isoformat()
        ci = forecast.get("confidence_interval", [0, 0])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO forecasts
                (issued_at, target_timestamp, horizon, model_used, predicted_aqi, ci_low, ci_high, peak_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    target_ts,
                    int(forecast["horizon"]),
                    str(forecast["model_used"]),
                    int(forecast["predicted_aqi"]),
                    int(ci[0]),
                    int(ci[1]),
                    forecast.get("peak_probability"),
                ),
            )

    def get_alerts(self, latest_forecast: dict | None = None, limit: int = 10) -> list[dict]:
        """Generate chronological alerts from observed high-AQI spikes and peak forecasts."""
        history = self.get_historical_readings(hours=168)
        alerts = []

        if latest_forecast and (
            latest_forecast.get("predicted_aqi", 0) >= 200
            or latest_forecast.get("is_peak_warning")
        ):
            alerts.append(
                {
                    "id": "alert-live-forecast",
                    "type": "forecast_high",
                    "message": (
                        f"Forecast AQI ≥ 200 ({latest_forecast['model_used']}) — "
                        f"Very Unhealthy expected at t+{latest_forecast['horizon']}h"
                    ),
                    "aqiValue": int(latest_forecast["predicted_aqi"]),
                    "timestamp": datetime.now(VN_TZ).isoformat(),
                }
            )

        spikes = [p for p in history if p["aqi"] >= 165]
        for idx, pt in enumerate(reversed(spikes[-limit:])):
            aqi_val = int(pt["aqi"])
            if aqi_val >= 300:
                a_type = "hazardous"
                msg = f"AQI reached Hazardous level ({pt['source']})"
            elif aqi_val >= 200:
                a_type = "very_unhealthy"
                msg = f"AQI reached Very Unhealthy level ≥ 200 ({pt['source']})"
            else:
                a_type = "forecast_high"
                msg = f"Elevated Unhealthy AQI regime detected ({pt['source']})"
            alerts.append(
                {
                    "id": f"alert-obs-{idx}-{pt['timestamp']}",
                    "type": a_type,
                    "message": msg,
                    "aqiValue": aqi_val,
                    "timestamp": pt["timestamp"],
                }
            )

        return alerts[:limit]
