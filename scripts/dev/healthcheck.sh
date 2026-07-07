#!/usr/bin/env bash
# One-command workspace check. Always uses the project .venv, runs from repo root.
#   bash scripts/healthcheck.sh           # full check
#   bash scripts/healthcheck.sh --quick   # skip the training smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "[FAIL] .venv not found at $PY"
  echo "       ↳ create it and install deps — see SETUP.md"
  exit 1
fi

exec "$PY" "$ROOT/scripts/healthcheck.py" "$@"
