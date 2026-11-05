"""Simulate an ESP32 (PMS7003 + BME280) node sending real-time telemetry to the Backend.

Usage:
    # Send one telemetry packet immediately:
    .venv/bin/python scripts/simulate_esp32.py --once

    # Stream packets every 10 seconds:
    .venv/bin/python scripts/simulate_esp32.py --interval 10
"""

import argparse
import json
import random
import time
import urllib.request


def send_packet(server_url: str, pm25: float, pm10: float, temp: float, rh: float, pressure: float):
    payload = {
        "pm25": round(pm25, 1),
        "pm10": round(pm10, 1),
        "temperature": round(temp, 1),
        "relativeHumidity": round(rh, 1),
        "pressure": round(pressure, 1),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/api/telemetry",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Simulate ESP32 PMS7003 + BME280 telemetry.")
    parser.add_argument("--server", default="http://localhost:5000", help="Backend server URL")
    parser.add_argument("--pm25", type=float, default=58.5, help="PM2.5 reading (ug/m3)")
    parser.add_argument("--pm10", type=float, default=82.0, help="PM10 reading (ug/m3)")
    parser.add_argument("--temp", type=float, default=26.8, help="Temperature (C)")
    parser.add_argument("--rh", type=float, default=76.5, help="Relative Humidity (%)")
    parser.add_argument("--pressure", type=float, default=1009.8, help="Pressure (hPa)")
    parser.add_argument("--interval", type=int, default=15, help="Seconds between packets")
    parser.add_argument("--once", action="store_true", help="Send a single packet and exit")
    args = parser.parse_args()

    pm25, pm10, temp, rh, pres = args.pm25, args.pm10, args.temp, args.rh, args.pressure
    while True:
        res = send_packet(args.server, pm25, pm10, temp, rh, pres)
        obs = res.get("observation", {})
        fc = res.get("forecast_t1", {})
        print(
            f"[ESP32 -> Server] Ingested PM2.5={obs.get('pm25')} PM10={obs.get('pm10')} "
            f"=> Current AQI={obs.get('aqi')} | "
            f"Forecast (t+1h, {fc.get('model_used')}): AQI={fc.get('predicted_aqi')} "
            f"CI={fc.get('confidence_interval')} PeakProb={fc.get('peak_probability')}"
        )
        if args.once:
            break
        time.sleep(args.interval)
        pm25 = max(8.0, min(300.0, pm25 + random.uniform(-4.0, 4.5)))
        pm10 = max(pm25 * 1.15, pm10 + random.uniform(-5.0, 5.5))


if __name__ == "__main__":
    main()

