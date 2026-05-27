#!/usr/bin/env bash
# Direct, simple GPU benchmark - no fancy stuff
# Test all 4 KV configs, collect metrics, generate report

set -euo pipefail

BINARY="/home/build/ollama/build_optimized/ollama"
LOG_DIR="/home/build/ollama/tmp-logs/final_$(date +%s)"
mkdir -p "$LOG_DIR"

RESULTS="$LOG_DIR/results.txt"
PORT=11534

# Kill any hanging server
pkill -9 ollama 2>/dev/null || true
sleep 1

run_config() {
  local kv_config="$1"
  local test_name="$2"
  local model="$3"
  local ctx="$4"
  local pred="$5"
  local prompt="$6"
  local timeout_sec="$7"
  
  echo "[*] Testing: $kv_config / $test_name"
  
  # Start server
  export OLLAMA_HOST="127.0.0.1:$PORT"
  export OLLAMA_KV_CACHE_TYPE="$kv_config"
  export OLLAMA_FLASH_ATTENTION=1
  export OLLAMA_GPU_LAYERS=999
  
  "$BINARY" serve >"$LOG_DIR/server_${kv_config//\//_}_${test_name}.log" 2>&1 &
  SERVER_PID=$!
  echo "    Server PID: $SERVER_PID"
  sleep 3
  
  # Wait for ready
  local ready=0
  for i in {1..60}; do
    if curl -s "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  
  if [[ $ready -eq 0 ]]; then
    echo "    ERROR: Server did not start"
    kill $SERVER_PID 2>/dev/null || true
    return 1
  fi
  
  echo "    Server ready, running inference..."
  
  # Prepare payload
  cat > /tmp/p.json <<EOF
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
  
  # Run inference
  local resp_file="$LOG_DIR/resp_${kv_config//\//_}_${test_name}.json"
  if timeout "$timeout_sec" curl -sS --max-time "$timeout_sec" \
    "http://127.0.0.1:$PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d @/tmp/p.json \
    >"$resp_file" 2>/dev/null; then
    
    # Extract metrics
    python3 - "$resp_file" "$kv_config" "$test_name" "$model" "$ctx" "$pred" >> "$RESULTS" <<'PYEOF'
import json, sys
try:
  with open(sys.argv[1]) as f:
    d = json.load(f)
  status = "ok" if d.get("done") else "incomplete"
  pr_cnt = d.get("prompt_eval_count", 0) or 0
  pr_dur = (d.get("prompt_eval_duration", 0) or 0) / 1e9
  ev_cnt = d.get("eval_count", 0) or 0
  ev_dur = (d.get("eval_duration", 0) or 0) / 1e9
  tot_dur = (d.get("total_duration", 0) or 0) / 1e9
  pr_tps = pr_cnt / pr_dur if pr_dur > 0 else 0
  ev_tps = ev_cnt / ev_dur if ev_dur > 0 else 0
  resp = str(d.get("response", ""))[:40].replace("\t", " ").replace("\n", " ")
  print(f"{sys.argv[2]}\t{sys.argv[3]}\t{sys.argv[4]}\t{sys.argv[5]}\t{sys.argv[6]}\t{status}\t{pr_cnt}\t{pr_dur:.4f}\t{pr_tps:.2f}\t{ev_cnt}\t{ev_dur:.4f}\t{ev_tps:.2f}\t{tot_dur:.4f}\t{resp}")
except Exception as e:
  print(f"ERROR: {e}")
PYEOF
    
    echo "    Inference completed"
  else
    echo "    ERROR: Inference failed or timed out"
  fi
  
  # Cleanup
  kill $SERVER_PID 2>/dev/null || true
  sleep 2
}

echo "[+] GPU Benchmark: TurboQuant vs RotorQuant"
echo "[+] Log dir: $LOG_DIR"
echo ""

# Write header
cat > "$RESULTS" <<'HEADER'
kv_config	test_name	model	context	predict	status	prompt_tokens	prompt_time	prompt_tps	gen_tokens	gen_time	gen_tps	total_time	response_preview
HEADER

# Test all 4 configs
for kv_config in "q8_0/q8_0" "q8_0/turbo3_0" "q8_0/planar3" "q8_0/iso3"; do
  echo "[+] Config: $kv_config"
  
  # Small test
  run_config "$kv_config" "small" "granite4.1:3b" "2048" "64" \
    "Return exactly one lowercase word: apple" "240"
  
  sleep 3
  
  # Medium test
  run_config "$kv_config" "medium" "granite4.1:3b" "4096" "16" \
    "Explain quantum computing." "600"
  
  sleep 3
done

echo ""
echo "[+] Results saved to: $RESULTS"
cat "$RESULTS"
