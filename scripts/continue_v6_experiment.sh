#!/usr/bin/env bash
# 완료된 Simple을 보존하고 체크포인트가 없는 Reason 단계부터 다시 진행.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo '사용: bash scripts/continue_v6_experiment.sh <기존 modified 실행 폴더> [--dry-run]' >&2
  exit 2
fi
if [[ " $* " != *' --dry-run '* ]]; then
  exec 9>"$1/.continue.lock"
  flock -n 9 || { echo '이 실행은 이미 이어서 실행 중입니다.' >&2; exit 1; }
  source scripts/hf_auth.sh
  require_hf_token
fi
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
if [[ " $* " == *' --dry-run '* ]]; then
  exec .venv/bin/python scripts/continue_v6_experiment.py "$@"
fi
.venv/bin/python scripts/continue_v6_experiment.py "$@" \
  2>&1 | tee -a "$1/continue_$(date +%Y%m%d_%H%M%S).log"
