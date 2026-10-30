#!/usr/bin/env bash
# Optional auto-shutdown: stop the VM after IDLE_MIN minutes without a finished request.
# Run on your laptop while the SSH tunnel is open:   IDLE_MIN=20 ./idle_stop.sh
# It reads vLLM's Prometheus counter vllm:request_success_total from /metrics and calls the Nebius API to stop
# the VM (a Linux shutdown on the VM itself would keep billing). Also stops after MAX_HOURS no matter what.
set -euo pipefail
URL="${METRICS_URL:-http://localhost:8000/metrics}"
IDLE_MIN="${IDLE_MIN:-20}"
MAX_HOURS="${MAX_HOURS:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
start=$(date +%s); last_change=$start; last=""
while true; do
  now=$(date +%s)
  count=$(curl -s -m 10 "$URL" | awk '/^vllm:request_success_total/ {s+=$2} END {print s+0}' || echo "")
  if [ -n "$count" ] && [ "$count" != "$last" ]; then last="$count"; last_change=$now; fi
  idle=$(( (now - last_change) / 60 )); up=$(( (now - start) / 3600 ))
  echo "$(date +%H:%M) requests=$count idle=${idle}m"
  if [ "$idle" -ge "$IDLE_MIN" ] || [ "$up" -ge "$MAX_HOURS" ]; then
    echo "Idle ${idle} min (or ${MAX_HOURS} h cap): stopping the VM."; exec "$HERE/stop_vm.sh"
  fi
  sleep 60
done
