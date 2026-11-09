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
    profile = config.get('experiment_profile', 'c1_scene')
    if profile not in ('c1_scene', 'ablation'):
        raise ValueError('experiment_profile must be c1_scene or ablation')
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
    for key in ('train_on_responses_only', 'finetune_attention_modules'):
        if type(common.get(key, False)) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if profile != 'ablation' and common['train_on_responses_only'] is not False:
        raise ValueError('C1 uses the public CUDA full-text loss policy in both backends')
    if profile != 'ablation' and common.get('finetune_attention_modules', False):
        raise ValueError('Attention LoRA changes require experiment_profile=ablation')
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
    path = Path(path or DEFAULT_PATH)
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('Experiment config must be a mapping')
    return validate_config(data)


def training_run_settings(config, config_path):
    """Runtime defaults, with paths anchored to the config file, not the shell cwd."""
    run = copy.deepcopy(config.get('run', {}))
    paths = {'data_path', 'base_image_path', 'split_path', 'reasoning_source',
             'output_dir', 'resume_from_checkpoint'}
    allowed = paths | {'mode', 'use_wandb', 'wandb_project', 'wandb_run_name',
                       'exact_output_dir', 'max_steps'}
    if not isinstance(run, dict) or set(run) - allowed:
        raise ValueError('Unknown run settings; training hyperparameters belong in common/backends')
    if 'mode' in run and run['mode'] not in ('simple', 'reason'):
        raise ValueError('run.mode must be simple or reason')
    for key in ('use_wandb', 'exact_output_dir'):
        if key in run and type(run[key]) is not bool:
            raise ValueError(f'run.{key} must be a boolean')
    if 'max_steps' in run and (type(run['max_steps']) is not int or run['max_steps'] == 0 or run['max_steps'] < -1):
        raise ValueError('run.max_steps must be -1 or a positive integer')
    for key in paths & run.keys():
        if run[key] is None and key == 'resume_from_checkpoint':
            continue
        if not isinstance(run[key], str) or not run[key].strip():
            raise ValueError(f'run.{key} must be a nonempty path')
        if key == 'reasoning_source' and run[key] == 'original':
            continue
        path = Path(run[key]).expanduser()
        run[key] = str((Path(config_path).resolve().parent / path).resolve())
    return run


def backend_settings(config, backend):
    if backend == 'mlx' and config.get('experiment_profile') == 'ablation':
        raise ValueError('The ablation training profile currently supports CUDA only')
    if backend == 'mlx' and config['common']['batch_size'] != 1:
        raise ValueError('The MLX workflow currently requires batch_size=1; use configs/experiment.json')
    return {**config['common'], **config['backends'][backend]}


def vision_collator_settings(config):
    """Keep the actual CUDA loss mask tied to the recorded experiment settings."""
    settings = {
        'max_seq_length': config['max_seq_length'],
        'resize': (config['image_size'], config['image_size']),
        'train_on_responses_only': config['train_on_responses_only'],
    }
    if config['train_on_responses_only']:
        # These are the complete Qwen 2.5 chat role markers, not source-data tags.
        # In particular, <think> remains supervised with the assistant response.
        settings.update(instruction_part='<|im_start|>user\n',
                        response_part='<|im_start|>assistant\n')
    return settings


def validate_resume_checkpoint(config, output_dir, checkpoint):
    """Reject partial, foreign, changed-config and already completed resumes."""
    output_dir, checkpoint = Path(output_dir).resolve(), Path(checkpoint).resolve()
    if not checkpoint.is_relative_to(output_dir) or checkpoint.parent != output_dir:
        raise ValueError('Resume checkpoint must be a direct child of the requested output directory')
    if (output_dir / 'completed.json').exists():
        raise ValueError('Training is already complete; choose a new output directory for a new experiment')
    for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                 'adapter_config.json', 'adapter_model.safetensors'):
        if not (checkpoint / name).is_file():
            raise ValueError(f'Incomplete resume checkpoint: missing {name}')
    saved = json.loads((output_dir / 'training_config.json').read_text())
    for key in ('experiment_sha256', 'mode', 'max_steps'):
        if saved.get(key) != config.get(key):
            raise ValueError(f'Resume settings differ from saved training: {key}')
    if saved.get('reasoning_source_provenance', {'kind': 'original'}) != config.get('reasoning_source_provenance', {'kind': 'original'}):
        raise ValueError('Resume reasoning source changed')
    for key in ('data_path', 'base_image_path'):
        if Path(saved[key]).resolve() != Path(config[key]).resolve():
            raise ValueError(f'Resume input path differs from saved training: {key}')
    current_split = json.loads(Path(config['split_path']).read_text())
    saved_split = json.loads((output_dir / 'split_manifest.json').read_text())
    if current_split != saved_split:
        raise ValueError('Resume split differs from the saved training split')
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if not isinstance(state.get('global_step'), int) or state['global_step'] <= 0:
        raise ValueError('Resume checkpoint has no completed optimizer steps')
    return str(checkpoint)


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
