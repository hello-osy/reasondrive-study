#!/usr/bin/env bash
# Automatic data preparation and sequential M4/16GB workflow.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'Apple Silicon macOS 전용입니다.' >&2
  exit 1
fi
IMAGE_ROOT="${1:-$PWD/data/nuscenes}"
EPOCHS="${EPOCHS:-3}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-1}"
case "$GRAD_CHECKPOINT" in
  0) CHECKPOINT_OPTION=--no-grad_checkpoint ;;
  1) CHECKPOINT_OPTION=--grad_checkpoint ;;
  *) echo "GRAD_CHECKPOINT must be 0 or 1" >&2; exit 1 ;;
esac
if ! command -v uv >/dev/null; then
  command -v brew >/dev/null || { echo 'Homebrew 또는 uv를 먼저 설치하세요.' >&2; exit 1; }
  brew install uv
fi
if [[ ! -x .venv-mac/bin/python ]]; then
  uv venv --python 3.11 .venv-mac
fi
uv pip install --python .venv-mac/bin/python -r requirements-mac.txt
if ! java -version >/dev/null 2>&1; then
  command -v brew >/dev/null || { echo 'Java 17+를 설치하고 다시 실행하세요.' >&2; exit 1; }
  if [[ ! -x "$(brew --prefix)/opt/openjdk@17/bin/java" ]]; then
    brew install openjdk@17
  fi
  export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
  export PATH="$JAVA_HOME/bin:$PATH"
fi
java -version
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
PYTHON="$PWD/.venv-mac/bin/python"
RUN_DIR="outputs/mac_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
# JSON과 필요한 이미지 자동 다운로드·검증. 기존 파일은 재사용합니다.
"$PYTHON" scripts/download_data.py --image_root "$IMAGE_ROOT"
"$PYTHON" scripts/prepare_inputs.py --image_root "$IMAGE_ROOT" \
  --split_path "splits/drivelm.json" --report_path "$RUN_DIR/input_repair_report.json"
"$PYTHON" scripts/mac_pipeline.py prepare --image_root "$IMAGE_ROOT"
# 첫 학습에서 모델 가중치 약 3.07 GB 다운로드. 두 모델은 캐시를 공유합니다.
"$PYTHON" scripts/mac_pipeline.py train "$CHECKPOINT_OPTION" --mode reason --steps 12 --output_dir "$RUN_DIR/smoke"
"$PYTHON" - "$RUN_DIR/smoke/completed.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"시험 학습 MLX peak: {r['peak_memory_gb']:.2f} GiB (전체 프로세스 메모리는 더 큼)")
if r['peak_memory_gb'] > 9:
    raise SystemExit('16GB 기기의 여유 메모리가 부족할 수 있어 전체 학습을 중단했습니다. GRAD_CHECKPOINT=1로 실행하거나 설정을 줄여 재검증하세요.')
PY
"$PYTHON" scripts/mac_pipeline.py train "$CHECKPOINT_OPTION" --epochs "$EPOCHS" --mode simple --output_dir "$RUN_DIR/simple"
"$PYTHON" scripts/mac_pipeline.py train "$CHECKPOINT_OPTION" --epochs "$EPOCHS" --mode reason --output_dir "$RUN_DIR/reason"
"$PYTHON" scripts/mac_pipeline.py infer --model_path "$RUN_DIR/simple" --output_path "$RUN_DIR/simple.json"
"$PYTHON" scripts/mac_pipeline.py infer --model_path "$RUN_DIR/reason" --output_path "$RUN_DIR/reason.json"
"$PYTHON" scripts/evaluate.py --responses "simple=$RUN_DIR/simple.json" "reason=$RUN_DIR/reason.json" \
  --data_path data/drivelm.json --output_dir "$RUN_DIR/comparison"
echo "완료: $PWD/$RUN_DIR/comparison/comparison.md"
