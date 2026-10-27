#!/usr/bin/env bash
# ============================================
# 실험용 확장·설정 변경 — 파일 전체
# 보고서용 공개 코드 조건 / 수정본 조건의 Simple·Reason 실험 실행기.
# ============================================
set -euo pipefail
cd "$(dirname "$0")/.."
usage() {
  echo '사용: bash scripts/run_report_experiment.sh public_baseline|modified|both [이미지 경로] [--dry-run]'
  echo '프로젝트 루트 기준 기본 이미지: data/nuscenes'
}
PROFILE="${1:---help}"
case "$PROFILE" in
  -h|--help) usage; exit 0 ;;
  public_baseline|modified|both) shift ;;
  *) usage >&2; exit 2 ;;
esac
IMAGE_ROOT=data/nuscenes
DRY_RUN=0
IMAGE_SET=0
for ARG in "$@"; do
  case "$ARG" in
    --dry-run) DRY_RUN=1 ;;
    --*) usage >&2; exit 2 ;;
    *) [[ $IMAGE_SET == 0 ]] || { usage >&2; exit 2; }; IMAGE_ROOT="$ARG"; IMAGE_SET=1 ;;
  esac
done
if [[ $DRY_RUN == 1 ]]; then
  python3 scripts/report_experiment.py plan --profile "$PROFILE" --image_root "$IMAGE_ROOT"
  exit 0
fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo '두 조건의 비교 실험은 NVIDIA GPU가 있는 Ubuntu 24.04 x86_64에서 실행하세요.' >&2
  echo 'Mac 수정본 단독 실행은 bash scripts/run_mac.sh 를 사용합니다.' >&2
  exit 1
}
export REPORT_RESULTS_ROOT="${REPORT_RESULTS_ROOT:-outputs/report_experiments}"
source scripts/hf_auth.sh
require_hf_token
PROFILES=("$PROFILE")
[[ $PROFILE != both ]] || PROFILES=(public_baseline modified)
for CURRENT_PROFILE in "${PROFILES[@]}"; do
  REPORT_PROFILE="$CURRENT_PROFILE" bash scripts/run_ubuntu.sh "$IMAGE_ROOT"
  .venv/bin/python scripts/report_experiment.py summarize --root "$REPORT_RESULTS_ROOT"
done
