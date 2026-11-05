#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "============================================================"
echo " Starting Real-Time AIoT AQI Forecasting System (Hanoi)"
echo " - Backend API (OpenAQ 4946813 + ESP32 + 10 Models): :5000"
echo " - React Dashboard (Vite)                          : :5173"
echo "============================================================"

cleanup() {
    echo ""
    echo "Stopping services..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/python src/serving/server.py &
BACKEND_PID=$!

cd "$ROOT_DIR/frontend"
npm run dev

