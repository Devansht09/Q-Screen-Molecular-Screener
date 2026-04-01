#!/usr/bin/env bash
# Q-Screen — One-shot setup script
# Usage: chmod +x setup.sh && ./setup.sh

set -e
BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}"
echo "  ██████╗       ███████╗ ██████╗██████╗ ███████╗███████╗███╗   ██╗"
echo "  ██╔═══██╗     ██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝████╗  ██║"
echo "  ██║   ██║─────███████╗██║     ██████╔╝█████╗  █████╗  ██╔██╗ ██║"
echo "  ██║▄▄ ██║     ╚════██║██║     ██╔══██╗██╔══╝  ██╔══╝  ██║╚██╗██║"
echo "  ╚██████╔╝     ███████║╚██████╗██║  ██║███████╗███████╗██║ ╚████║"
echo "   ╚══▀▀═╝      ╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝"
echo -e "${RESET}"
echo -e "${BOLD}  Quantum Molecule Stability Screener — Setup${RESET}"
echo ""

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Step 1: Python deps ──────────────────────────────────────────────────────
echo -e "${GREEN}[1/4] Installing Python dependencies...${RESET}"
if conda env list | grep -q "qscreen"; then
  echo "  → conda env 'qscreen' already exists, activating..."
else
  conda create -y -n qscreen python=3.10
fi

# Detect if conda available
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate qscreen
  conda install -y -c conda-forge rdkit -q
  pip install fastapi "uvicorn[standard]" scikit-learn pandas numpy joblib httpx -q
else
  echo "  → conda not found, using pip..."
  pip install rdkit fastapi "uvicorn[standard]" scikit-learn pandas numpy joblib httpx -q
fi

echo "  ✓ Python dependencies installed"

# ── Step 2: Train model ──────────────────────────────────────────────────────
echo -e "${GREEN}[2/4] Training ML model...${RESET}"
cd "$ROOT/ml"
python train_model.py
echo "  ✓ Model saved: ml/qscreen_model.pkl"

# ── Step 3: Frontend ─────────────────────────────────────────────────────────
echo -e "${GREEN}[3/4] Installing frontend dependencies...${RESET}"
cd "$ROOT/frontend"
if command -v npm &>/dev/null; then
  npm install -q
  echo "  ✓ npm packages installed"
else
  echo "  ✗ npm not found — install Node.js from https://nodejs.org"
  exit 1
fi

# ── Step 4: Launch ───────────────────────────────────────────────────────────
echo -e "${GREEN}[4/4] Starting servers...${RESET}"
echo ""
echo -e "${BOLD}  Backend  →  http://localhost:8000${RESET}"
echo -e "${BOLD}  Frontend →  http://localhost:3000${RESET}"
echo -e "${BOLD}  API Docs →  http://localhost:8000/docs${RESET}"
echo ""

# Start backend in background
cd "$ROOT/backend"
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo -e "${GREEN}  ✓ Q-Screen is running! Press Ctrl+C to stop.${RESET}"
echo ""

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '  Q-Screen stopped.'" EXIT
wait
