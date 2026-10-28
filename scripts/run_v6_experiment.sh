#!/usr/bin/env bash
# V6 표 8에 필요한 scene-split Simple/Reason만 새로 학습·평가한다.
set -euo pipefail
cd "$(dirname "$0")/.."
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/experiment-cuda-batch30.json}"
exec bash scripts/run_report_experiment.sh modified "$@"
