"""Real-time AQI Backend Server (Port 5000) for React Dashboard, OpenAQ v3, and ESP32 IoT Node.

Usage:
    .venv/bin/python src/serving/server.py
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.serving.ingestor import RealtimeDataStore, load_env_file
from src.serving.predictor import MODEL_BENCHMARKS_T1, RealtimePredictor


class RealtimeAppContext:
    def __init__(self):
        load_env_file()
        self.store = RealtimeDataStore()
        self.store.ensure_warm_start(hours=240)
        self.predictor = RealtimePredictor()
        self.last_sync_result: dict = {}
        self._lock = threading.Lock()

    def sync_now(self) -> dict:
        with self._lock:
            self.store.ensure_warm_start(hours=240)
            res = self.store.sync_openaq_and_weather(hours=72)
            self.last_sync_result = res
            return res


CTX: RealtimeAppContext | None = None


class AQIRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json({"ok": True}, status=200)

    def do_GET(self):
        global CTX
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/api/health":
                current = CTX.store.get_current_reading()
                self._send_json(
                    {
                        "status": "ok",
                        "openaq_location_id": CTX.store.openaq_location_id,
                        "station_name": "Số 1 đường Giải Phóng - phường Bạch Mai - ĐHBK",
                        "loaded_models_t1": [
                            k for k in CTX.predictor._models_by_horizon.get(1, {}) if not k.startswith("_")
                        ],
                        "loaded_models_t6": [
                            k for k in CTX.predictor._models_by_horizon.get(6, {}) if not k.startswith("_")
                        ],
                        "latest_observation": current,
                        "last_sync": CTX.last_sync_result,
                    }
                )
                return

            if path == "/api/current":
                current = CTX.store.get_current_reading()
                if current is None:
                    CTX.store.ensure_warm_start(hours=168)
                    current = CTX.store.get_current_reading()
                self._send_json(current)
                return

            if path == "/api/forecast":
                horizon = int(query.get("horizon", ["1"])[0])
                model = query.get("model", ["Best available"])[0]
                history = CTX.store.get_historical_readings(hours=48)
                if len(history) < 48:
                    CTX.store.ensure_warm_start(hours=168)
                    history = CTX.store.get_historical_readings(hours=48)
                forecast = CTX.predictor.predict(history, horizon=horizon, model_name=model)
                self._send_json(forecast)
                return

            if path in ("/api/historical", "/api/history"):
                hours = int(query.get("hours", ["168"])[0])
                model = query.get("model", ["XGBoost"])[0]
                lookback = hours + 49
                raw_history = CTX.store.get_historical_readings(hours=lookback)
                if len(raw_history) < lookback:
                    CTX.store.ensure_warm_start(hours=max(240, lookback))
                    raw_history = CTX.store.get_historical_readings(hours=lookback)
                enriched = CTX.predictor.attach_rolling_predictions(
                    raw_history, horizon=1, model_name=model
                )
                self._send_json(enriched[-hours:])
                return

            if path in ("/api/model-comparison", "/api/models"):
                self._send_json(MODEL_BENCHMARKS_T1)
                return

            if path == "/api/alerts":
                history = CTX.store.get_historical_readings(hours=48)
                latest_fc = (
                    CTX.predictor.predict(history, horizon=1, model_name="Two-Stage Regressor")
                    if len(history) >= 48
                    else None
                )
                alerts = CTX.store.get_alerts(latest_forecast=latest_fc, limit=10)
                self._send_json(alerts)
                return

            self._send_json({"error": f"Not found: {path}"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self):
        global CTX
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(raw_body) if raw_body.strip() else {}

            if path == "/api/sync":
                sync_res = CTX.sync_now()
                self._send_json(sync_res)
                return

            if path in ("/api/telemetry", "/api/v1/telemetry"):
                obs = CTX.store.ingest_esp32_telemetry(payload)
                history = CTX.store.get_historical_readings(hours=48)
                forecast_t1 = CTX.predictor.predict(history, horizon=1, model_name="Best available")
                CTX.store.record_forecast(forecast_t1)
                self._send_json(
                    {
                        "status": "ingested",
                        "observation": obs,
                        "forecast_t1": forecast_t1,
                    },
                    status=201,
                )
                return

            self._send_json({"error": f"Not found: {path}"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt, *args):
        sys.stdout.write(f"[AQI-API] {self.address_string()} - {fmt % args}\n")


def _background_sync_worker(ctx: RealtimeAppContext, interval_sec: int = 900):
    """Sync OpenAQ & weather on startup and every `interval_sec` seconds (default 15m)."""
    while True:
        try:
            res = ctx.sync_now()
            print(
                f"[OpenAQ Sync] station={res.get('openaq_location_id')} "
                f"openaq_hours={res.get('openaq_records_fetched')} "
                f"weather_hours={res.get('weather_records_fetched')} "
                f"updated={res.get('rows_updated')}"
            )
            if res.get("errors"):
                print(f"[OpenAQ Sync Notice] {'; '.join(res['errors'])}")
        except Exception as exc:
            print(f"[OpenAQ Sync Error] {exc}")
        time.sleep(interval_sec)


def run_server(host: str | None = None, port: int | None = None):
    global CTX
    load_env_file()
    host = host or os.getenv("HOST", "0.0.0.0")
    port = int(port or os.getenv("PORT", "5000"))

    print("Initializing Real-Time AQI Forecasting Backend...")
    CTX = RealtimeAppContext()

    sync_thread = threading.Thread(target=_background_sync_worker, args=(CTX, 900), daemon=True)
    sync_thread.start()

    server = ThreadingHTTPServer((host, port), AQIRequestHandler)
    print(f"Real-Time AQI Backend listening on http://{host}:{port}")
    print(f"  - Health & status : http://localhost:{port}/api/health")
    print(f"  - Current reading : http://localhost:{port}/api/current")
    print(f"  - Forecast (t+1/6): http://localhost:{port}/api/forecast?horizon=1&model=XGBoost")
    print(f"  - ESP32 Telemetry : POST http://localhost:{port}/api/telemetry")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    run_server()

