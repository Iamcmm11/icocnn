#!/usr/bin/env bash
set -euo pipefail

cd /home/cmm/icocnn

WAIT_PID="${1:-}"
PYTHON_BIN="/home/cmm/miniconda3/envs/icocnn/bin/python"
LOG_DIR="IFAN_Edge/outputs/stage3/logs"
mkdir -p "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/saf_lite_phase1_queue_$(date +%Y%m%d_%H%M%S).log"
echo "$(date --iso-8601=seconds) queue_start wait_pid=${WAIT_PID:-none}" >> "${QUEUE_LOG}"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    echo "$(date --iso-8601=seconds) waiting_for_pid=${WAIT_PID}" >> "${QUEUE_LOG}"
    sleep 300
  done
  echo "$(date --iso-8601=seconds) wait_pid_done=${WAIT_PID}" >> "${QUEUE_LOG}"
fi

for cfg in \
  IFAN_Edge/configs/stage3_saf_lite_4of8_phase1.toml \
  IFAN_Edge/configs/stage3_saf_lite_3of8_phase1.toml \
  IFAN_Edge/configs/stage3_saf_lite_2of8_phase1.toml
do
  name="$(basename "${cfg}" .toml)"
  run_log="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "$(date --iso-8601=seconds) run_start cfg=${cfg} log=${run_log}" >> "${QUEUE_LOG}"
  "${PYTHON_BIN}" IFAN_Edge/scripts/train_stage3_ifan.py --config "${cfg}" --device cuda > "${run_log}" 2>&1
  rc=$?
  echo "$(date --iso-8601=seconds) run_done cfg=${cfg} rc=${rc} log=${run_log}" >> "${QUEUE_LOG}"
  if [[ "${rc}" -ne 0 ]]; then
    exit "${rc}"
  fi
done

echo "$(date --iso-8601=seconds) queue_complete" >> "${QUEUE_LOG}"
