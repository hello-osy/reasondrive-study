#!/usr/bin/env bash
# Ubuntu 24.04 / amd64 / NVIDIA (RTX 5070 Ti) sequential workflow.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo 'Ubuntu 24.04 amd64 (x86_64) 전용입니다.' >&2; exit 1;
}
source /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 24.04 ]] || {
  echo 'Ubuntu 24.04에서 실행하세요.' >&2; exit 1;
}
EPOCHS="${EPOCHS:-3}"
[[ $EPOCHS =~ ^[1-9][0-9]*$ ]] || { echo 'EPOCHS는 양의 정수여야 합니다.' >&2; exit 1; }
IMAGE_ROOT="${1:-$PWD/data/nuscenes}"
MODEL_ID="${MODEL_ID:-unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit}"
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
uv pip install --python "$PYTHON" --torch-backend=auto -r requirements-cuda.txt
uv pip check --python "$PYTHON"
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
RUN_DIR=$(mktemp -d "$PWD/outputs_ubuntu_$(date +%Y%m%d_%H%M%S)_XXXXXX")
uv pip freeze --python "$PYTHON" > "$RUN_DIR/requirements-resolved.txt"
nvidia-smi > "$RUN_DIR/nvidia-smi.txt"
echo "실행 결과: $RUN_DIR"
"$PYTHON" - <<'PY'
from unsloth import FastVisionModel
import torch
import bitsandbytes as bnb
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
PY
"$PYTHON" scripts/download_data.py --image_root "$IMAGE_ROOT"
"$PYTHON" scripts/prepare_inputs.py --image_root "$IMAGE_ROOT" \
  --split_path "$RUN_DIR/split_manifest.json" --report_path "$RUN_DIR/input_repair_report.json"
COMMON=(--model_id "$MODEL_ID" --data_path data/drivelm.json
        --base_image_path "$IMAGE_ROOT" --split_path "$RUN_DIR/split_manifest.json"
        --epochs "$EPOCHS" --batch_size 1 --gradient_accumulation_steps 12
        --max_seq_length 2048 --no-use_wandb --exact_output_dir)
# One optimizer step = 12 microsteps. Failure stops the whole pipeline.
"$PYTHON" scripts/vlm-finetuning.py "${COMMON[@]}" --mode reason --max_steps 1 --output_dir "$RUN_DIR/smoke"
for MODE in simple reason; do
  "$PYTHON" scripts/vlm-finetuning.py "${COMMON[@]}" --mode "$MODE" --output_dir "$RUN_DIR/$MODE"
done
for MODE in simple reason; do
  "$PYTHON" scripts/eval_finetuned_models.py --model_path "$RUN_DIR/$MODE" \
    --data_path data/drivelm.json --base_image_path "$IMAGE_ROOT" --output_path "$RUN_DIR/$MODE.json"
done
"$PYTHON" scripts/evaluate.py --responses "simple=$RUN_DIR/simple.json" "reason=$RUN_DIR/reason.json" \
  --data_path data/drivelm.json --output_dir "$RUN_DIR/comparison"
echo "완료: $RUN_DIR/comparison/comparison.md"
