#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/hf_auth.sh
require_hf_token
docker run --gpus all \
  -v "$PWD":/workspace/ \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY="${DISPLAY:-}" \
  -e HF_TOKEN \
  -e HF_HUB_DISABLE_IMPLICIT_TOKEN \
  --ipc=host \
  --network=host \
  --rm \
  --name reasondrive \
  -it qwenllm/qwenvl:2.5-cu121 bash
