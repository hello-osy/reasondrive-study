#!/usr/bin/env bash
# Resume the interrupted public profile, then continue the modified profile.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo '사용: bash scripts/resume_report_experiment.sh <기존 public_baseline 실행 폴더> [--batch-size 8] [--public-only]' >&2
  exit 2
fi
RUN_DIR=$(cd "$1" && pwd)
shift
PUBLIC_ONLY=0
RESUME_OPTIONS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --public-only) PUBLIC_ONLY=1; shift ;;
    --batch-size)
      [[ $# -ge 2 && $2 =~ ^[1-9][0-9]*$ ]] || { echo '--batch-size에는 양의 정수가 필요합니다.' >&2; exit 2; }
      RESUME_OPTIONS=(--resume_batch_size "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ -x .venv/bin/python && -f "$RUN_DIR/public_settings.json" && -f "$RUN_DIR/experiment.json" ]] || {
  echo '기존 Python 환경과 public_baseline 실행 설정이 필요합니다.' >&2; exit 1;
}
exec 9>"$RUN_DIR/.resume.lock"
flock -n 9 || { echo '이 실행 폴더를 이미 재개 중입니다.' >&2; exit 1; }
source scripts/hf_auth.sh
require_hf_token
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
mapfile -t INPUT_PATHS < <(.venv/bin/python - "$RUN_DIR/simple/training_result.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
print(record['data_path'])
print(record['image_root'])
PY
)
[[ ${#INPUT_PATHS[@]} == 2 ]] || { echo '기존 데이터 경로를 읽을 수 없습니다.' >&2; exit 1; }
.venv/bin/python scripts/report_experiment.py public_baseline \
  --resume "${RESUME_OPTIONS[@]}" --run_dir "$RUN_DIR" --config "$RUN_DIR/experiment.json" \
  --data_path "${INPUT_PATHS[0]}" --image_root "${INPUT_PATHS[1]}" \
  2>&1 | tee -a "$RUN_DIR/resume.log"
RESULTS_ROOT=$(dirname "$(dirname "$RUN_DIR")")
.venv/bin/python scripts/report_experiment.py summarize --root "$RESULTS_ROOT"
if [[ $PUBLIC_ONLY == 0 ]]; then
  EXPERIMENT_CONFIG="$RUN_DIR/experiment.json" REPORT_RESULTS_ROOT="$RESULTS_ROOT" \
    bash scripts/run_report_experiment.sh modified "${INPUT_PATHS[1]}"
fi
