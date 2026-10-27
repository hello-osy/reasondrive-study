# ============================================
# 실험용 확장·설정 변경: 공통 조건 및 재현성 기록
# ============================================
"""실험용 확장·설정 변경: 공통 조건, 고정 모델 revision, 실제 실행 출처를 기록한다.

설정 읽기와 단위 테스트는 모델을 다운로드하지 않는다. resolve_model_source는
사용자가 학습/추론을 실행했을 때만 호출하며 고정 snapshot에서 모델과 processor를 읽는다.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'experiment.json'
SHA = re.compile(r'^[0-9a-f]{40}$')


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_config(config):
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported experiment schema')
    common = config['common']
    for key in ('epochs', 'batch_size', 'gradient_accumulation_steps', 'max_seq_length',
                'image_size', 'max_new_tokens', 'early_stopping_patience', 'lora_r', 'lora_alpha'):
        if type(common[key]) is not int or common[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(common['seed']) is not int or common['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(common['grad_checkpoint']) is not bool:
        raise ValueError('grad_checkpoint must be a boolean')
    if common['image_size'] != 224:
        raise ValueError('The shared input formatter currently supports image_size=224 only')
    if common['batch_size'] != 1:
        raise ValueError('The paired CUDA/MLX workflow currently requires batch_size=1')
    if common['train_on_responses_only'] is not False:
        raise ValueError('C1 uses the public CUDA full-text loss policy in both backends')
    if common['evaluation_strategy'] != 'epoch' or common['do_sample'] is not False:
        raise ValueError('C1 uses epoch validation and deterministic generation')
    if not 0 < common['val_size'] < 1 or not 0 < common['test_size'] < 1 or common['val_size'] + common['test_size'] >= 1:
        raise ValueError('Invalid validation/test fractions')
    if common['learning_rate'] <= 0 or common['weight_decay'] < 0 or not 0 <= common['lora_dropout'] < 1:
        raise ValueError('Invalid optimizer/LoRA settings')
    for name in ('cuda', 'mlx'):
        backend = config['backends'][name]
        if not backend['model_id'] or not SHA.fullmatch(backend['model_revision']):
            raise ValueError(f'{name}: an explicit model ID and immutable 40-digit revision are required')
        if type(backend['warmup_steps']) is not int or backend['warmup_steps'] < 0:
            raise ValueError(f'{name}: warmup_steps must be a nonnegative integer')
    mlx = config['backends']['mlx']
    if mlx['optimizer'] != 'adamw' or mlx['scheduler'] != 'constant' or mlx['warmup_steps'] != 0:
        raise ValueError('The MLX implementation supports AdamW/constant/no-warmup only')
    return config


def load_experiment(path=None):
    return validate_config(json.loads(Path(path or DEFAULT_PATH).read_text()))


def backend_settings(config, backend):
    return {**config['common'], **config['backends'][backend]}


def override_experiment(config, backend, overrides):
    result = copy.deepcopy(config)
    for key, value in overrides.items():
        if value is None:
            continue
        if key in result['common']:
            result['common'][key] = value
        elif key in result['backends'][backend]:
            result['backends'][backend][key] = value
        else:
            raise ValueError(f'Unknown experiment setting: {key}')
    if overrides.get('model_id') and overrides['model_id'] != config['backends'][backend]['model_id'] and not overrides.get('model_revision'):
        raise ValueError('Changing model_id also requires --model_revision')
    return validate_config(result)


def resolve_model_source(model_id, revision):
    """고정 snapshot의 로컬 경로를 반환하여 모델과 processor의 출처를 일치시킨다."""
    if not SHA.fullmatch(revision or ''):
        raise ValueError('An immutable model revision is required before loading weights')
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(model_id, revision=revision, allow_patterns=[
        '*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.tiktoken'])).resolve()
    if snapshot.name != revision:
        raise ValueError(f'Cannot verify resolved snapshot revision: {snapshot}')
    return str(snapshot), {'model_id': model_id, 'requested_revision': revision,
                           'resolved_revision': snapshot.name, 'processor_revision': snapshot.name,
                           'snapshot_path': str(snapshot)}


def runtime_metadata(backend):
    packages = {}
    for package in ('torch', 'torchvision', 'transformers', 'unsloth', 'unsloth_zoo', 'trl',
                    'bitsandbytes', 'peft', 'datasets', 'mlx', 'mlx-vlm', 'huggingface-hub'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {'backend': backend, 'python': sys.version, 'platform': platform.platform(),
              'packages': packages}
    root = DEFAULT_PATH.parent.parent
    for key, args in [('code_commit', ['rev-parse', 'HEAD']), ('working_tree', ['status', '--porcelain'])]:
        try:
            result[key] = subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            result[key] = None
    torch = sys.modules.get('torch')
    if backend == 'cuda' and torch is not None and torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                      torch_cuda=torch.version.cuda, device_index=torch.cuda.current_device())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_PATH))
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', choices=['cuda', 'mlx'], required=True)
    parser.add_argument('--model_id')
    parser.add_argument('--model_revision')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--grad_checkpoint', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = override_experiment(load_experiment(args.config), args.backend,
                                {key: getattr(args, key) for key in ('model_id', 'model_revision', 'epochs', 'grad_checkpoint')})
    from data_utils import write_json
    write_json(args.output, config)
    print(f'고정 실행 설정: {args.output} (SHA256 {config_hash(config)})')


if __name__ == '__main__':
    main()
