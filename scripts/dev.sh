#!/usr/bin/env bash
# Run frontend and backend dev servers simultaneously.
# Usage: ./scripts/dev.sh

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cleanup() {
    echo "Shutting down..."
    kill 0
    wait
}
trap cleanup SIGINT SIGTERM

# Backend (Flask + SocketIO)
echo "Starting backend on http://localhost:5001 ..."
cd "$ROOT_DIR"
python -m playground.game_viewer.app &

# Frontend (Vite)
echo "Starting frontend ..."
cd "$ROOT_DIR/playground/frontend"
npm run dev &

wait
