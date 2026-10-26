#!/usr/bin/env bash
# ============================================
# 논문 기반 재구현: Instruct / Simple / Reason 공통 조건 실행.
# 실험용 확장·설정 변경: Ubuntu CUDA 자동화, 설정 snapshot, 의존성 재설치 경로.
# ============================================
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo 'Ubuntu 24.04 amd64 (x86_64) 전용입니다.' >&2; exit 1;
}
source /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 24.04 ]] || {
  echo 'Ubuntu 24.04에서 실행하세요.' >&2; exit 1;
}
IMAGE_ROOT="${1:-$PWD/data/nuscenes}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
command -v nvidia-smi >/dev/null && nvidia-smi || {
  echo 'NVIDIA 드라이버를 설치하고 재부팅한 뒤 nvidia-smi부터 확인하세요.' >&2; exit 1;
}
command -v uv >/dev/null || { echo '문서의 uv 설치 명령을 먼저 실행하세요.' >&2; exit 1; }
java -version || { echo 'sudo apt install openjdk-17-jre-headless 를 먼저 실행하세요.' >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
PYTHON="$PWD/.venv/bin/python"
if [[ -n ${DEPENDENCY_SNAPSHOT:-} ]]; then
  uv pip sync --python "$PYTHON" --torch-backend=auto "$DEPENDENCY_SNAPSHOT"
else
  uv pip install --python "$PYTHON" --torch-backend=auto -r requirements-cuda.txt
fi
uv pip check --python "$PYTHON"
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
RUN_DIR=$(mktemp -d "$PWD/outputs_ubuntu_$(date +%Y%m%d_%H%M%S)_XXXXXX")
uv pip freeze --python "$PYTHON" > "$RUN_DIR/requirements-resolved.txt"
uv --version > "$RUN_DIR/uv-version.txt"
nvidia-smi > "$RUN_DIR/nvidia-smi.txt"
CONFIG_OPTIONS=(--backend cuda --config "${EXPERIMENT_CONFIG:-configs/experiment.json}" --output "$RUN_DIR/experiment.json")
[[ -z ${EPOCHS:-} ]] || CONFIG_OPTIONS+=(--epochs "$EPOCHS")
[[ -z ${MODEL_ID:-} ]] || CONFIG_OPTIONS+=(--model_id "$MODEL_ID")
[[ -z ${MODEL_REVISION:-} ]] || CONFIG_OPTIONS+=(--model_revision "$MODEL_REVISION")
"$PYTHON" scripts/experiment_config.py "${CONFIG_OPTIONS[@]}"
RUN_CONFIG="$RUN_DIR/experiment.json"
echo "실행 결과: $RUN_DIR"
"$PYTHON" - "$RUN_DIR" <<'PY'
import sys
from unsloth import FastVisionModel
import torch
import bitsandbytes as bnb
from scripts.experiment_config import runtime_metadata
from scripts.data_utils import write_json
if not torch.cuda.is_available():
    raise SystemExit('CUDA를 사용할 수 없습니다. NVIDIA 드라이버와 PyTorch 설치를 확인하세요.')
cap = torch.cuda.get_device_capability()
cuda = tuple(map(int, (torch.version.cuda or '0.0').split('.')[:2]))
print('GPU:', torch.cuda.get_device_name(), 'capability:', cap,
      'torch:', torch.__version__, 'CUDA:', torch.version.cuda)
if cap[0] >= 12 and cuda < (12, 8):
    raise SystemExit('RTX 50 Blackwell에는 CUDA 12.8 이상의 PyTorch wheel이 필요합니다.')
# Actually execute GPU kernels, including the quantized layer used for training.
x = torch.randn(32, 64, device='cuda', dtype=torch.float16, requires_grad=True)
layer = bnb.nn.Linear4bit(64, 64, quant_type='nf4', compute_dtype=torch.float16).cuda()
y = layer(x)
y.float().sum().backward()
torch.cuda.synchronize()
if not torch.isfinite(y).all():
    raise SystemExit('4bit CUDA 사전 점검 실패')
print('CUDA / bitsandbytes 4bit 실행 확인')
write_json(sys.argv[1] + '/environment.json', runtime_metadata('cuda'))
PY
"$PYTHON" scripts/download_data.py --image_root "$IMAGE_ROOT"
"$PYTHON" scripts/prepare_inputs.py --image_root "$IMAGE_ROOT" \
  --config "$RUN_CONFIG" --split_path "$RUN_DIR/split_manifest.json" --report_path "$RUN_DIR/input_repair_report.json"
COMMON=(--config "$RUN_CONFIG" --data_path data/drivelm.json
        --base_image_path "$IMAGE_ROOT" --split_path "$RUN_DIR/split_manifest.json"
        --no-use_wandb --exact_output_dir)
# One optimizer step = 12 microsteps. Failure stops the whole pipeline.
"$PYTHON" scripts/vlm-finetuning.py "${COMMON[@]}" --mode reason --max_steps 1 --output_dir "$RUN_DIR/smoke"
for MODE in simple reason; do
  "$PYTHON" scripts/vlm-finetuning.py "${COMMON[@]}" --mode "$MODE" --output_dir "$RUN_DIR/$MODE"
done
for MODE in simple reason; do
  "$PYTHON" scripts/eval_finetuned_models.py --config "$RUN_CONFIG" --model_path "$RUN_DIR/$MODE" \
    --data_path data/drivelm.json --base_image_path "$IMAGE_ROOT" --output_path "$RUN_DIR/$MODE.json"
done
"$PYTHON" scripts/eval_finetuned_models.py --config "$RUN_CONFIG" --instruct \
  --split_path "$RUN_DIR/split_manifest.json" --data_path data/drivelm.json \
  --base_image_path "$IMAGE_ROOT" --output_path "$RUN_DIR/instruct.json"
"$PYTHON" scripts/evaluate.py --responses "instruct=$RUN_DIR/instruct.json" "simple=$RUN_DIR/simple.json" "reason=$RUN_DIR/reason.json" \
  --data_path data/drivelm.json --output_dir "$RUN_DIR/comparison"
echo "완료: $RUN_DIR/comparison/comparison.md"
