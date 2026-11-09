#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python scripts/local_reasoning.py "$@"
fi
if command -v uv >/dev/null; then
  exec uv run --python 3.11 --with huggingface-hub scripts/local_reasoning.py "$@"
fi
echo '기존 .venv 또는 uv가 필요합니다. 먼저 프로젝트 환경을 준비하세요.' >&2
exit 1
