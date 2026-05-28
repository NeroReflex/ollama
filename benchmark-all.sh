#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BUILD_LOG_DIR="$ROOT_DIR/tmp-logs"
RUN_DIR="$BUILD_LOG_DIR/bench_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

BINARY="$ROOT_DIR/build_optimized/ollama"
RESULTS_TSV="$RUN_DIR/results.tsv"
REPORT_MD="$ROOT_DIR/BENCHMARK.md"
SERVER_PORT=11534

KV_CONFIGS=(
  "q8_0/q8_0"
  "q8_0/turbo3_0"
  "q8_0/planar3"
  "q8_0/iso3"
)

QWEN_MODEL="${QWEN_MODEL:-}"
if [[ -z "$QWEN_MODEL" ]]; then
  if [[ -e "$HOME/.ollama/models/manifests/registry.ollama.ai/library/qwen3.6/27b" ]]; then
    QWEN_MODEL="qwen3.6:27b"
  else
    QWEN_MODEL="qwen3.6:35b"
  fi
fi

QWEN_KV_CONFIGS=(
  "turbo2_0/turbo2_0"
  "turbo3_0/turbo3_0"
  "turbo4_0/turbo4_0"
  "q8_0/turbo2_0"
)

QWEN_CTX="${QWEN_CTX:-131072}"
# NEVER set qwen coherence tests to tiny num_predict values (for example: 1).
# Doing so truncates before final content and makes coherence checks meaningless.
QWEN_PREDICT="${QWEN_PREDICT:-192}"
QWEN_BATCH="${QWEN_BATCH:-1}"
QWEN_THINK="${QWEN_THINK:-true}"
QWEN_TIMEOUT="${QWEN_TIMEOUT:-900}"
QWEN_MIN_GEN_TOKENS="${QWEN_MIN_GEN_TOKENS:-96}"
QWEN_MIN_RESPONSE_WORDS="${QWEN_MIN_RESPONSE_WORDS:-50}"
QWEN_PROMPT="${QWEN_PROMPT:-Think through the request, then provide a coherent 120-160 word explanation of why apples are commonly used in examples, with clear structure and no bullet points.}"

QWEN_TEST_NAMES=(
  "turbo2_turbo2"
  "turbo3_turbo3"
  "turbo4_turbo4"
  "q8_0_turbo2"
)

TEST_NAMES=("small" "medium")
TEST_MODELS=("granite4.1:3b" "granite4.1:3b")
TEST_CTX=("2048" "4096")
TEST_PRED=("16" "16")
TEST_PROMPTS=(
  "Explain quantum computing in one paragraph."
  "Explain quantum computing in one paragraph."
)
TEST_TIMEOUT=("300" "900")

SERVER_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  pkill -9 ollama 2>/dev/null || true
}

trap cleanup_server EXIT

log() {
  printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"
}

require_tool() {
  local tool="$1"
  command -v "$tool" >/dev/null 2>&1 || {
    echo "missing required tool: $tool" >&2
    exit 1
  }
}

check_gpu() {
  require_tool nvidia-smi
  nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi cannot access GPU" >&2
    exit 1
  }
}

build_fresh() {
  log "Building with Makefile.build (full build)"
  make -f Makefile.build build JOBS="${JOBS:-$(nproc)}" > "$RUN_DIR/build.log" 2>&1
  if [[ ! -x "$BINARY" ]]; then
    echo "expected binary not found: $BINARY" >&2
    exit 1
  fi
}

wait_ready() {
  local tries=90
  local i
  for ((i=1; i<=tries; i++)); do
    if curl -s "http://127.0.0.1:${SERVER_PORT}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_server() {
  local kv="$1"
  local slog="$2"

  cleanup_server

  export OLLAMA_HOST="127.0.0.1:${SERVER_PORT}"
  export OLLAMA_FLASH_ATTENTION=1
  export OLLAMA_GPU_LAYERS=999
  export OLLAMA_KV_CACHE_TYPE="$kv"

  "$BINARY" serve > "$slog" 2>&1 &
  SERVER_PID=$!

  wait_ready
}

run_one() {
  local kv="$1"
  local test_idx="$2"

  local tname="${TEST_NAMES[$test_idx]}"
  local model="${TEST_MODELS[$test_idx]}"
  local ctx="${TEST_CTX[$test_idx]}"
  local pred="${TEST_PRED[$test_idx]}"
  local prompt="${TEST_PROMPTS[$test_idx]}"
  local timeout_s="${TEST_TIMEOUT[$test_idx]}"

  local kv_safe="${kv//\//_}"
  local server_log="$RUN_DIR/server_${kv_safe}_${tname}.log"
  local payload="$RUN_DIR/payload_${kv_safe}_${tname}.json"
  local warmup_payload="$RUN_DIR/payload_${kv_safe}_${tname}_warmup.json"
  local resp="$RUN_DIR/response_${kv_safe}_${tname}.json"
  local errf="$RUN_DIR/curl_${kv_safe}_${tname}.stderr"

  log "Config=$kv test=$tname start"

  if ! start_server "$kv" "$server_log"; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$kv" "$tname" "$model" "$ctx" "$pred" "server_start_failed" \
      "0" "0" "0" "0" "0" "0" "0" "0" "0" "server_not_ready" >> "$RESULTS_TSV"
    return
  fi

  cat > "$payload" <<EOF
{
  "model": "$model",
  "prompt": "$prompt",
  "stream": false,
  "keep_alive": "0s",
  "options": {
    "num_predict": $pred,
    "num_ctx": $ctx,
    "temperature": 0
  }
}
EOF

  cat > "$warmup_payload" <<EOF
{
  "model": "$model",
  "prompt": "warm up",
  "stream": false,
  "keep_alive": "0s",
  "options": {
    "num_predict": 8,
    "num_ctx": $ctx,
    "temperature": 0
  }
}
EOF

  curl -sS --max-time 120 \
    "http://127.0.0.1:${SERVER_PORT}/api/generate" \
    -H "Content-Type: application/json" \
    -d @"$warmup_payload" > /dev/null 2>&1 || true

  : > "$errf"
  if ! timeout "$timeout_s" curl -sS --max-time "$timeout_s" \
      "http://127.0.0.1:${SERVER_PORT}/api/generate" \
      -H "Content-Type: application/json" \
      -d @"$payload" > "$resp" 2> "$errf"; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$kv" "$tname" "$model" "$ctx" "$pred" "request_failed" \
      "0" "0" "0" "0" "0" "0" "0" "0" "0" "curl_failed" >> "$RESULTS_TSV"
    return
  fi

  # Some runs can return done=false without timing out; retry once for stable metrics.
  if ! python3 - "$resp" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
except Exception:
    raise SystemExit(1)
if not d.get('done'):
    raise SystemExit(1)
if (d.get('prompt_eval_count') or 0) <= 0:
    raise SystemExit(1)
raise SystemExit(0)
PY
  then
    timeout "$timeout_s" curl -sS --max-time "$timeout_s" \
      "http://127.0.0.1:${SERVER_PORT}/api/generate" \
      -H "Content-Type: application/json" \
      -d @"$payload" > "$resp" 2> "$errf" || true
  fi

  python3 - "$resp" "$server_log" "$kv" "$tname" "$model" "$ctx" "$pred" >> "$RESULTS_TSV" <<'PY'
import json, os, re, sys

resp_path, slog_path, kv, tname, model, ctx, pred = sys.argv[1:]

def has(pat, txt):
    return re.search(pat, txt, re.IGNORECASE) is not None

try:
    with open(resp_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
except Exception as e:
    print("\t".join([kv, tname, model, ctx, pred, "response_parse_failed", "0", "0", "0", "0", "0", "0", "0", "0", "0", f"parse_error:{e}"]))
    raise SystemExit

try:
    with open(slog_path, 'r', encoding='utf-8', errors='ignore') as f:
        slog = f.read()
except Exception:
    slog = ""

prompt_count = int(d.get("prompt_eval_count") or 0)
prompt_dur_ns = int(d.get("prompt_eval_duration") or 0)
eval_count = int(d.get("eval_count") or 0)
eval_dur_ns = int(d.get("eval_duration") or 0)
total_dur_ns = int(d.get("total_duration") or 0)

def tps(count, dur_ns):
    return (count / (dur_ns / 1e9)) if dur_ns > 0 else 0.0

prompt_tps = tps(prompt_count, prompt_dur_ns)
gen_tps = tps(eval_count, eval_dur_ns)

cuda_detected = int(has(r"msg=\"inference compute\".*library=CUDA", slog) and has(r"llama_prepare_model_devices: using device CUDA0", slog))
cuda_tensors = int(has(r"load_tensors:\s+CUDA0 model buffer size", slog))
load_failed = int(has(r"Load failed", slog) or has(r"signal arrived during cgo execution", slog) or has(r"cannot run the operation \(SET_ROWS\)", slog))

done = bool(d.get("done"))
error = d.get("error")

response_text = str(d.get("response") or "")
response_word_count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", response_text))
qwen_tests = {"turbo2_turbo2", "turbo3_turbo3", "turbo4_turbo4", "q8_0_turbo2"}
min_gen_tokens = int(os.environ.get("QWEN_MIN_GEN_TOKENS", "96"))
min_response_words = int(os.environ.get("QWEN_MIN_RESPONSE_WORDS", "50"))
coherent_qwen = True
if tname in qwen_tests:
  coherent_qwen = eval_count >= min_gen_tokens and response_word_count >= min_response_words

status = "ok" if done and not error and cuda_detected and not load_failed and coherent_qwen else "failed"
errtxt = "" if error is None else str(error)
if not errtxt and tname in qwen_tests and not coherent_qwen:
  errtxt = (
    f"incoherent_output eval_count={eval_count} words={response_word_count} "
    f"required_eval={min_gen_tokens} required_words={min_response_words}"
  )

print("\t".join([
    kv, tname, model, ctx, pred, status,
    str(prompt_count), f"{prompt_dur_ns/1e9:.6f}", f"{prompt_tps:.2f}",
    str(eval_count), f"{eval_dur_ns/1e9:.6f}", f"{gen_tps:.2f}",
    f"{total_dur_ns/1e9:.6f}", str(cuda_detected), str(cuda_tensors), errtxt.replace("\t", " ")[:300]
]))
PY

  log "Config=$kv test=$tname done"
}

run_case() {
  local kv="$1"
  local model="$2"
  local tname="$3"
  local ctx="$4"
  local pred="$5"
  local prompt="$6"
  local timeout_s="$7"
  local warmup="${8:-1}"
  local batch_size="${9:-}"

  local kv_safe="${kv//\//_}"
  local model_safe="${model//[:\/.]/_}"
  local server_log="$RUN_DIR/server_${model_safe}_${kv_safe}_${tname}.log"
  local payload="$RUN_DIR/payload_${model_safe}_${kv_safe}_${tname}.json"
  local warmup_payload="$RUN_DIR/payload_${model_safe}_${kv_safe}_${tname}_warmup.json"
  local resp="$RUN_DIR/response_${model_safe}_${kv_safe}_${tname}.json"
  local errf="$RUN_DIR/curl_${model_safe}_${kv_safe}_${tname}.stderr"

  log "Config=$kv model=$model test=$tname start"

  if ! start_server "$kv" "$server_log"; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$kv" "$tname" "$model" "$ctx" "$pred" "server_start_failed" \
      "0" "0" "0" "0" "0" "0" "0" "0" "0" "server_not_ready" >> "$RESULTS_TSV"
    return
  fi

  local batch_line=""
  if [[ -n "$batch_size" ]]; then
    batch_line=",
    \"num_batch\": $batch_size"
  fi

  cat > "$payload" <<EOF
{
  "model": "$model",
  "prompt": "$prompt",
  "think": ${QWEN_THINK},
  "stream": false,
  "keep_alive": "0s",
  "options": {
    "num_predict": $pred,
    "num_ctx": $ctx,
    "temperature": 0${batch_line}
  }
}
EOF

  cat > "$warmup_payload" <<EOF
{
  "model": "$model",
  "prompt": "warm up",
  "stream": false,
  "keep_alive": "0s",
  "options": {
    "num_predict": 8,
    "num_ctx": $ctx,
    "temperature": 0
  }
}
EOF

  if [[ "$warmup" == "1" ]]; then
    curl -sS --max-time 120 \
      "http://127.0.0.1:${SERVER_PORT}/api/generate" \
      -H "Content-Type: application/json" \
      -d @"$warmup_payload" > /dev/null 2>&1 || true
  fi

  : > "$errf"
  if ! timeout "$timeout_s" curl -sS --max-time "$timeout_s" \
      "http://127.0.0.1:${SERVER_PORT}/api/generate" \
      -H "Content-Type: application/json" \
      -d @"$payload" > "$resp" 2> "$errf"; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$kv" "$tname" "$model" "$ctx" "$pred" "request_failed" \
      "0" "0" "0" "0" "0" "0" "0" "0" "0" "curl_failed" >> "$RESULTS_TSV"
    return
  fi

  if ! python3 - "$resp" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
except Exception:
    raise SystemExit(1)
if not d.get('done'):
    raise SystemExit(1)
if (d.get('prompt_eval_count') or 0) <= 0:
    raise SystemExit(1)
raise SystemExit(0)
PY
  then
    timeout "$timeout_s" curl -sS --max-time "$timeout_s" \
      "http://127.0.0.1:${SERVER_PORT}/api/generate" \
      -H "Content-Type: application/json" \
      -d @"$payload" > "$resp" 2> "$errf" || true
  fi

  python3 - "$resp" "$server_log" "$kv" "$tname" "$model" "$ctx" "$pred" >> "$RESULTS_TSV" <<'PY'
import json, re, sys

resp_path, slog_path, kv, tname, model, ctx, pred = sys.argv[1:]

def has(pat, txt):
    return re.search(pat, txt, re.IGNORECASE) is not None

try:
    with open(resp_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
except Exception as e:
    print("\t".join([kv, tname, model, ctx, pred, "response_parse_failed", "0", "0", "0", "0", "0", "0", "0", "0", "0", f"parse_error:{e}"]))
    raise SystemExit

try:
    with open(slog_path, 'r', encoding='utf-8', errors='ignore') as f:
        slog = f.read()
except Exception:
    slog = ""

prompt_count = int(d.get("prompt_eval_count") or 0)
prompt_dur_ns = int(d.get("prompt_eval_duration") or 0)
eval_count = int(d.get("eval_count") or 0)
eval_dur_ns = int(d.get("eval_duration") or 0)
total_dur_ns = int(d.get("total_duration") or 0)

def tps(count, dur_ns):
    return (count / (dur_ns / 1e9)) if dur_ns > 0 else 0.0

prompt_tps = tps(prompt_count, prompt_dur_ns)
gen_tps = tps(eval_count, eval_dur_ns)

cuda_detected = int(has(r"msg=\"inference compute\".*library=CUDA", slog) and has(r"llama_prepare_model_devices: using device CUDA0", slog))
cuda_tensors = int(has(r"load_tensors:\s+CUDA0 model buffer size", slog))
load_failed = int(has(r"Load failed", slog) or has(r"signal arrived during cgo execution", slog) or has(r"cannot run the operation \(SET_ROWS\)", slog))

done = bool(d.get("done"))
error = d.get("error")

status = "ok" if done and not error and cuda_detected and not load_failed else "failed"
errtxt = "" if error is None else str(error)

print("\t".join([
    kv, tname, model, ctx, pred, status,
    str(prompt_count), f"{prompt_dur_ns/1e9:.6f}", f"{prompt_tps:.2f}",
    str(eval_count), f"{eval_dur_ns/1e9:.6f}", f"{gen_tps:.2f}",
    f"{total_dur_ns/1e9:.6f}", str(cuda_detected), str(cuda_tensors), errtxt.replace("\t", " ")[:300]
]))
PY

  log "Config=$kv model=$model test=$tname done"
}

run_qwen36_turbo_benchmark() {
  local i
  local kv
  local tname

  for i in "${!QWEN_KV_CONFIGS[@]}"; do
    kv="${QWEN_KV_CONFIGS[$i]}"
    tname="${QWEN_TEST_NAMES[$i]}"
    run_case "$kv" "$QWEN_MODEL" "$tname" "$QWEN_CTX" "$QWEN_PREDICT" \
      "$QWEN_PROMPT" "$QWEN_TIMEOUT" "0" "$QWEN_BATCH"
  done
}

generate_report() {
  QWEN_MODEL="$QWEN_MODEL" QWEN_CTX="$QWEN_CTX" QWEN_PREDICT="$QWEN_PREDICT" QWEN_BATCH="$QWEN_BATCH" QWEN_THINK="$QWEN_THINK" QWEN_PROMPT="$QWEN_PROMPT" QWEN_MIN_GEN_TOKENS="$QWEN_MIN_GEN_TOKENS" QWEN_MIN_RESPONSE_WORDS="$QWEN_MIN_RESPONSE_WORDS" python3 - "$RESULTS_TSV" "$REPORT_MD" <<'PY'
import csv
import datetime as dt
import os
import sys

results_path, report_path = sys.argv[1:]
with open(results_path, 'r', encoding='utf-8') as f:
  rows = list(csv.DictReader(f, delimiter='\t'))

now = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')

def fmtf(x):
  try:
    return f"{float(x):.2f}"
  except Exception:
    return "0.00"

def by_test(name):
  return [r for r in rows if r['test_name'] == name]

def row_map(test_rows):
  return {r['kv_config']: r for r in test_rows}

cfg_order = ["q8_0/q8_0", "q8_0/turbo3_0", "q8_0/planar3", "q8_0/iso3"]
qwen_order = ["turbo2_turbo2", "turbo3_turbo3", "turbo4_turbo4", "q8_0_turbo2"]
qwen_model = os.environ.get('QWEN_MODEL', 'qwen3.6:27b')
qwen_ctx = os.environ.get('QWEN_CTX', '131072')
qwen_pred = os.environ.get('QWEN_PREDICT', '1')
qwen_batch = os.environ.get('QWEN_BATCH', '1')
qwen_think = os.environ.get('QWEN_THINK', 'false')
qwen_prompt = os.environ.get('QWEN_PROMPT', 'N/A')
qwen_min_gen = os.environ.get('QWEN_MIN_GEN_TOKENS', '96')
qwen_min_words = os.environ.get('QWEN_MIN_RESPONSE_WORDS', '50')

small = row_map(by_test('small'))
medium = row_map(by_test('medium'))

all_ok = all(r.get('status') == 'ok' for r in rows)

lines = []
lines.append('# KV Cache Benchmark Report')
lines.append('')
lines.append(f'**Date (UTC):** {now}')
lines.append('**Model:** granite4.1:3b')
lines.append('**Binary:** build_optimized/ollama (built via `make -f Makefile.build build`)')
lines.append('**Runtime Flags:** OLLAMA_FLASH_ATTENTION=1, OLLAMA_GPU_LAYERS=999')
lines.append('**KV Policy:** key fixed at q8_0, value varied')
lines.append('')
lines.append('## Executive Summary')
lines.append('')
lines.append('| Config | Small Prompt tok/s | Small Gen tok/s | Medium Prompt tok/s | Medium Gen tok/s | GPU Validated | Status |')
lines.append('|---|---:|---:|---:|---:|---|---|')
for cfg in cfg_order:
  s = small.get(cfg)
  m = medium.get(cfg)
  if not s or not m:
    lines.append(f'| {cfg} | - | - | - | - | no | missing |')
    continue
  gpu_ok = (s.get('cuda_detected') == '1' and s.get('cuda_tensors') == '1' and m.get('cuda_detected') == '1' and m.get('cuda_tensors') == '1')
  st = 'ok' if s.get('status') == 'ok' and m.get('status') == 'ok' else 'failed'
  lines.append(f"| {cfg} | {fmtf(s.get('prompt_tps'))} | {fmtf(s.get('gen_tps'))} | {fmtf(m.get('prompt_tps'))} | {fmtf(m.get('gen_tps'))} | {'yes' if gpu_ok else 'no'} | {st} |")

lines.append('')
lines.append('## Detailed Results')
lines.append('')
lines.append('| Config | Test | Status | Prompt Tokens | Prompt s | Prompt tok/s | Gen Tokens | Gen s | Gen tok/s | Total s | CUDA Detected | CUDA Tensors | Error |')
lines.append('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|')
for cfg in cfg_order:
  for tname in ('small', 'medium'):
    r = small.get(cfg) if tname == 'small' else medium.get(cfg)
    if not r:
      lines.append(f'| {cfg} | {tname} | missing | - | - | - | - | - | - | - | - | - | - |')
      continue
    lines.append(f"| {cfg} | {tname} | {r.get('status')} | {r.get('prompt_tokens')} | {float(r.get('prompt_time_s', 0)):.4f} | {fmtf(r.get('prompt_tps'))} | {r.get('gen_tokens')} | {float(r.get('gen_time_s', 0)):.4f} | {fmtf(r.get('gen_tps'))} | {float(r.get('total_time_s', 0)):.4f} | {r.get('cuda_detected')} | {r.get('cuda_tensors')} | {r.get('error', '')} |")

lines.append('')
lines.append('## qwen3.6 turbo coherence')
lines.append('')
lines.append(f'**Model:** {qwen_model}')
lines.append(f'**Context:** {qwen_ctx}')
lines.append(f'**Predict:** {qwen_pred}')
lines.append(f'**Batch:** {qwen_batch}')
lines.append(f'**Think:** {qwen_think}')
lines.append(f'**Min Gen Tokens:** {qwen_min_gen}')
lines.append(f'**Min Response Words:** {qwen_min_words}')
lines.append(f'**Prompt:** {qwen_prompt}')
lines.append('')
lines.append('| Test | KV Config | Status | Prompt Tokens | Prompt s | Prompt tok/s | Gen Tokens | Gen s | Gen tok/s | Total s | CUDA Validated | Error |')
lines.append('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|')
for tname in qwen_order:
  r = next((row for row in rows if row['test_name'] == tname), None)
  if not r:
    lines.append(f'| {tname} | - | missing | - | - | - | - | - | - | - | no | - |')
    continue
  gpu_ok = (r.get('cuda_detected') == '1' and r.get('cuda_tensors') == '1')
  lines.append(f"| {tname} | {r.get('kv_config')} | {r.get('status')} | {r.get('prompt_tokens')} | {float(r.get('prompt_time_s', 0)):.4f} | {fmtf(r.get('prompt_tps'))} | {r.get('gen_tokens')} | {float(r.get('gen_time_s', 0)):.4f} | {fmtf(r.get('gen_tps'))} | {float(r.get('total_time_s', 0)):.4f} | {'yes' if gpu_ok else 'no'} | {r.get('error', '')} |")

lines.append('')
lines.append('## Interpretation')
lines.append('')
if all_ok:
  lines.append('- All four baseline configurations completed successfully with CUDA execution evidence in server logs.')
else:
  lines.append('- One or more baseline configurations failed; see Detailed Results and Error column for exact failure points.')
lines.append('- Prompt tok/s corresponds to prefill throughput.')
lines.append('- Gen tok/s corresponds to decode throughput.')
lines.append('- qwen3.6 turbo pairs are reported separately below because they use a different model and KV layout.')
lines.append('')
lines.append('## Method')
lines.append('')
lines.append('- Single unified script: `benchmark-all.sh`')
lines.append('- Build phase: `make -f Makefile.build build`')
lines.append('- For each KV config and test size: fresh server start, single generate request, metrics parsed from response JSON.')
lines.append('- GPU validation required: CUDA inference log line and CUDA model tensor placement log line.')
lines.append('')
lines.append('## Log Artifacts')
lines.append('')
lines.append(f'- Raw benchmark TSV: {results_path}')
lines.append('- Per-run server and response logs are in the same run directory under `tmp-logs`.')

with open(report_path, 'w', encoding='utf-8') as f:
  f.write('\n'.join(lines) + '\n')
PY
}

main() {
  require_tool curl
  require_tool python3
  check_gpu

  log "Run directory: $RUN_DIR"

  build_fresh

  printf "kv_config\ttest_name\tmodel\tcontext\tpredict\tstatus\tprompt_tokens\tprompt_time_s\tprompt_tps\tgen_tokens\tgen_time_s\tgen_tps\ttotal_time_s\tcuda_detected\tcuda_tensors\terror\n" > "$RESULTS_TSV"

  local kv
  local i
  for kv in "${KV_CONFIGS[@]}"; do
    for i in "${!TEST_NAMES[@]}"; do
      run_one "$kv" "$i"
    done
  done

  run_qwen36_turbo_benchmark

  cleanup_server

  generate_report
  log "Benchmark complete"
  log "Results: $RESULTS_TSV"
  log "Report: $REPORT_MD"
}

main "$@"
