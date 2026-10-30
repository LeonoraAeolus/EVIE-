#!/usr/bin/env bash
# Runs ON the GPU VM (installed by create_vm.sh as a systemd service; you can also run it by hand).
# Installs vLLM into a venv on first start, then serves OpenMath-Nemotron as an OpenAI-compatible API.
set -euo pipefail
MODEL="${OPENMATH_MODEL:-nvidia/OpenMath-Nemotron-7B}"
PORT="${PORT:-8000}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"          # 127.0.0.1 = reachable only through an SSH tunnel (safer)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"      # model supports 131,072; 16k fits prompt + 8k derivation with room
GPU_UTIL="${GPU_UTIL:-0.90}"
VENV=/opt/openmath/venv

if [ ! -x "$VENV/bin/vllm" ]; then
  sudo apt-get update -y && sudo apt-get install -y python3-venv python3-dev
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install vllm
fi

ARGS=(serve "$MODEL"
  --served-model-name "$MODEL"
  --host "$BIND_HOST" --port "$PORT"
  --dtype bfloat16                      # weights ship in bf16 (config.json torch_dtype)
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --generation-config vllm)             # the repo's generation_config.json sets no sampling; Tesla Lab sends its own
if [ -n "${OPENMATH_API_KEY:-}" ]; then ARGS+=(--api-key "$OPENMATH_API_KEY"); fi

exec "$VENV/bin/vllm" "${ARGS[@]}"
