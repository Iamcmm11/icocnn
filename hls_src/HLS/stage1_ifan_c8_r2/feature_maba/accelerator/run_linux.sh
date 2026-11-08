#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-cosim}"
shift || true
PYTHON="${PYTHON:-python3}"

if ! command -v vitis_hls >/dev/null 2>&1 && [[ "$MODE" != "native" ]]; then
  echo "vitis_hls was not found on PATH. Source the AMD settings64.sh for Vitis HLS 2024.2." >&2
  exit 2
fi

if [[ -n "${MABA_DATA_DIR:-}" ]]; then
  DATA_DIR="$MABA_DATA_DIR"
elif [[ -d "$SCRIPT_DIR/artifacts/data_hls" ]]; then
  DATA_DIR="$SCRIPT_DIR/artifacts/data_hls"
else
  DATA_DIR="$SCRIPT_DIR/testdata/rtl_smoke"
fi

exec "$PYTHON" "$SCRIPT_DIR/run.py" "$MODE" \
  --variants "${MABA_VARIANTS:-${MABA_VARIANT:-2}}" \
  --tiles "${MABA_TILES:-${MABA_TILE:-16}}" \
  --pars "${MABA_PARS:-${MABA_PAR:-1}}" \
  --frames "${MABA_FRAMES:-6}" \
  --fifo "${MABA_FIFO:-8}" \
  --rtl-limit "${MABA_RTL_LIMIT:-3}" \
  --timeout "${MABA_TIMEOUT:-14400}" \
  --data "$DATA_DIR" "$@"
