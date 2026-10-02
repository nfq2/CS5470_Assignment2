#!/bin/bash
# CS5470 HW2: run client.sh for several CFS intervals.
# Starts and stops its own server (server.sh) for every interval,
# so do NOT run server.sh yourself while this script runs.
#
# Usage (from the assignment_2 directory):  bash sweep.sh
# Output: results_3.2/{env.txt, p<interval>_r<rep>.txt, server_p<interval>.log}
set -euo pipefail

INTERVALS=${INTERVALS:-"0 5 10 20 40"}
REPS=${REPS:-1}
OUT=${OUT:-results_3.2}
mkdir -p "$OUT"
{ hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0,1; } > "$OUT/env.txt"

gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | sort -n | tail -1; }
STARTED=0
stop_server() {
  [ "$STARTED" = 1 ] || return 0      # only stop a server this script started
  pkill -f vllm.entrypoints.openai.api_server || true
  for _ in $(seq 60); do [ "$(gpu_mem)" -lt 1000 ] && break; sleep 2; done
  STARTED=0
}
trap stop_server EXIT

for p in $INTERVALS; do
  if curl -sf localhost:8000/health >/dev/null || [ "$(gpu_mem)" -gt 1000 ]; then
    echo "ERROR: GPUs or port 8000 already in use. Stop other servers first." >&2; exit 1
  fi

  echo "=== CFS interval $p ==="
  rm -f vllm_server.log
  CFS_INTERVAL=$p bash server.sh > "$OUT/server_p${p}.stdout" 2>&1 &
  STARTED=1
  until curl -sf localhost:8000/health >/dev/null; do
    kill -0 $! 2>/dev/null || { echo "ERROR: server failed; see $OUT/server_p${p}.stdout" >&2; exit 1; }
    sleep 5
  done

  # make sure the installed scheduler.py reads CFS_INTERVAL
  grep -q "CFS interval: $p\$" vllm_server.log || {
    echo "ERROR: server did not log 'CFS interval: $p'. Copy your scheduler.py into vLLM." >&2; exit 1; }
  grep -q "# cuda blocks: 1024," vllm_server.log || {
    echo "ERROR: GPU KV cache is not 1024 blocks. Do not modify server.sh." >&2; exit 1; }

  for r in $(seq "$REPS"); do
    bash client.sh > "$OUT/p${p}_r${r}.txt" 2>&1
    grep "P99 TTFT" "$OUT/p${p}_r${r}.txt" || true
  done

  stop_server
  cp vllm_server.log "$OUT/server_p${p}.log"
done
echo "Done. Results in $OUT"
