#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${1:-/tmp/ollama-install}"
BIN="${INSTALL_DIR}/bin/ollama"
LOG_DIR="${INSTALL_DIR}/gpu-test-logs"
SUMMARY="${LOG_DIR}/summary.txt"
UNLOAD_WAIT_SECS="${UNLOAD_WAIT_SECS:-180}"
VRAM_RECOVERY_MARGIN_MIB="${VRAM_RECOVERY_MARGIN_MIB:-256}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-}"

mkdir -p "${LOG_DIR}"
: > "${SUMMARY}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: missing executable: ${BIN}" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found" >&2
  exit 1
fi

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' '
}

has_running_models() {
  local ps_output
  ps_output="$(${BIN} ps 2>/dev/null || true)"
  awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }' <<<"${ps_output}"
}

port_in_use() {
  local port="$1"
  ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"
}

pick_server_port() {
  local candidate

  if [[ -n "${SERVER_PORT}" ]]; then
    if port_in_use "${SERVER_PORT}"; then
      echo "ERROR: requested SERVER_PORT=${SERVER_PORT} is already in use" | tee -a "${SUMMARY}"
      return 1
    fi
    return 0
  fi

  for candidate in 11534 11535 11536 11537 11538 11539 11540 11541 11542 11543 11544; do
    if ! port_in_use "${candidate}"; then
      SERVER_PORT="${candidate}"
      return 0
    fi
  done

  echo "ERROR: no free port found in test range 11534-11544" | tee -a "${SUMMARY}"
  return 1
}

wait_for_server_ready() {
  for _ in $(seq 1 30); do
    if "${BIN}" list >/dev/null 2>&1; then
      return 0
    fi

    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      if grep -q "address already in use" "${CURRENT_SERVER_LOG}" 2>/dev/null; then
        echo "ERROR: server bind collision on ${OLLAMA_HOST}" | tee -a "${SUMMARY}"
      else
        echo "ERROR: server exited before becoming ready (host ${OLLAMA_HOST})" | tee -a "${SUMMARY}"
      fi
      return 1
    fi

    sleep 1
  done

  return 1
}

wait_for_runner_exit() {
  for _ in $(seq 1 "${UNLOAD_WAIT_SECS}"); do
    if ! has_running_models; then
      return 0
    fi
    sleep 1
  done

  return 1
}

wait_for_vram_recovery() {
  local baseline="$1"
  local threshold=$((baseline + VRAM_RECOVERY_MARGIN_MIB))
  local current

  for _ in $(seq 1 "${UNLOAD_WAIT_SECS}"); do
    current="$(gpu_used_mib)"
    if (( current <= threshold )); then
      return 0
    fi
    sleep 1
  done

  return 1
}

start_server() {
  local server_log="$1"

  CURRENT_SERVER_LOG="${server_log}"
  : > "${server_log}"
  PATH="${INSTALL_DIR}/bin:${PATH}" \
  OLLAMA_HOST="${OLLAMA_HOST}" \
  OLLAMA_DEBUG=1 \
  OLLAMA_FLASH_ATTENTION=1 \
  OLLAMA_KV_CACHE_TYPE=turbo3_0 \
  OLLAMA_KEEP_ALIVE=0 \
  OLLAMA_LIBRARY_PATH="${INSTALL_DIR}/lib/ollama:${INSTALL_DIR}/lib/ollama/cuda_v13" \
  "${BIN}" serve > "${server_log}" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_server_ready; then
    echo "ERROR: server did not become ready on ${OLLAMA_HOST}" | tee -a "${SUMMARY}"
    return 1
  fi

  return 0
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}

echo "== GPU baseline ==" | tee -a "${SUMMARY}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv | tee -a "${SUMMARY}"
BASELINE_VRAM="$(gpu_used_mib)"
echo "baseline_vram=${BASELINE_VRAM}MiB recovery_margin=${VRAM_RECOVERY_MARGIN_MIB}MiB" | tee -a "${SUMMARY}"

if ! pick_server_port; then
  exit 1
fi

export OLLAMA_HOST="http://${SERVER_HOST}:${SERVER_PORT}"
echo "test_host=${OLLAMA_HOST}" | tee -a "${SUMMARY}"

SERVER_PID=""
CURRENT_SERVER_LOG=""
trap 'stop_server' EXIT

LIST_SERVER_LOG="${LOG_DIR}/list-server.log"
if ! start_server "${LIST_SERVER_LOG}"; then
  exit 1
fi

# Gather model names exactly as reported by ollama list.
mapfile -t MODELS < <("${BIN}" list | awk 'NR>1 {print $1}')

stop_server

if ! wait_for_vram_recovery "${BASELINE_VRAM}"; then
  echo "ERROR: VRAM did not recover after listing models" | tee -a "${SUMMARY}"
  exit 1
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No local models found." | tee -a "${SUMMARY}"
  exit 0
fi

echo "== Models to test (${#MODELS[@]}) ==" | tee -a "${SUMMARY}"
printf '%s\n' "${MODELS[@]}" | tee -a "${SUMMARY}"

for model in "${MODELS[@]}"; do
  SAFE_NAME="${model//[:\/]/_}"
  MODEL_LOG="${LOG_DIR}/${SAFE_NAME}.log"
  SERVER_LOG="${LOG_DIR}/${SAFE_NAME}.server.log"
  echo "" | tee -a "${SUMMARY}"
  echo "== Testing ${model} ==" | tee -a "${SUMMARY}"

  if ! start_server "${SERVER_LOG}"; then
    exit 1
  fi

  BEFORE="$(gpu_used_mib)"

  set +e
  timeout 180 curl -sS "${OLLAMA_HOST}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model}\",\"prompt\":\"Respond with one short sentence.\",\"stream\":false,\"keep_alive\":\"0s\",\"options\":{\"num_predict\":32}}" \
    > "${MODEL_LOG}" 2>&1
  RC=$?
  set -e

  AFTER_REQUEST="$(gpu_used_mib)"
  PEAK_DELTA=$((AFTER_REQUEST - BEFORE))

  RUNNER_EXIT_RESULT="OK"
  if ! wait_for_runner_exit; then
    RUNNER_EXIT_RESULT="TIMEOUT"
  fi

  stop_server

  VRAM_RECOVERY_RESULT="OK"
  if ! wait_for_vram_recovery "${BASELINE_VRAM}"; then
    VRAM_RECOVERY_RESULT="TIMEOUT"
  fi

  AFTER_UNLOAD="$(gpu_used_mib)"
  RECOVERY_DELTA=$((AFTER_UNLOAD - BASELINE_VRAM))

  if [[ ${RC} -eq 0 ]]; then
    if grep -q '"error"' "${MODEL_LOG}"; then
      RESULT="FAIL_API"
    else
      RESULT="PASS"
    fi
  elif [[ ${RC} -eq 124 ]]; then
    RESULT="TIMEOUT"
  else
    RESULT="FAIL(${RC})"
  fi

  # Look for clear GPU evidence in server log near this model load.
  GPU_EVIDENCE=$(grep -E "FlashAttention|flash_attn|KvCacheType|turbo3_0|load_tensors|CUDA0|CUDA|gpu|layers\.offload|offload" "${SERVER_LOG}" | tail -n 40 || true)

  echo "result=${RESULT} gpu_mem_before=${BEFORE}MiB gpu_mem_after_request=${AFTER_REQUEST}MiB peak_delta=${PEAK_DELTA}MiB gpu_mem_after_unload=${AFTER_UNLOAD}MiB recovery_delta=${RECOVERY_DELTA}MiB runner_exit=${RUNNER_EXIT_RESULT} vram_recovery=${VRAM_RECOVERY_RESULT}" | tee -a "${SUMMARY}"
  if [[ -n "${GPU_EVIDENCE}" ]]; then
    echo "gpu_evidence=YES" | tee -a "${SUMMARY}"
    echo "-- gpu evidence tail --" >> "${SUMMARY}"
    echo "${GPU_EVIDENCE}" >> "${SUMMARY}"
  else
    echo "gpu_evidence=NO" | tee -a "${SUMMARY}"
  fi

  echo "server_log=${SERVER_LOG}" | tee -a "${SUMMARY}"

done

echo "" | tee -a "${SUMMARY}"
echo "== Final GPU state ==" | tee -a "${SUMMARY}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv | tee -a "${SUMMARY}"

echo "Summary written to ${SUMMARY}"
