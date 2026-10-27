# ============================================
# 실험용 확장·설정 변경 — 파일 전체
# 고정한 공개 원본을 실행하는 CUDA 전용 연결부. 원본의 분할·실패 처리·길이·sampling을 유지한다.
# Simple 모드와 API 인수명 호환만 추가하며, 적용 내용을 실행 artifact에 기록한다.
# ============================================
"""Run the preserved public ReasonDrive code, with an explicitly labeled Simple extension.

The parent experiment runner owns data selection, model snapshot pinning and noGPT scoring.
Inference consumes ALL supplied JSON/JSONL rows in their given order, without splitting them.
No model library is imported when this module is imported or --help is requested.
"""
import argparse
import ast
import contextlib
import copy
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'public_reasondrive'
UPSTREAM_COMMIT = '326caecafedad1b9dbc156ebd77c8dc0eb998740'
SOURCE_FILES = ('vlm-finetuning.py', 'eval_finetuned_models.py', 'LICENSE')
PROMPT_REPLACEMENTS = (
    ('Analyze the scene and reason through driving decisions carefully.', 'Analyze the scene carefully.'),
    ('Analyze the following driving scenario and provide reasoning:',
     'Analyze the following driving scenario and provide the final answer:'),
    ('Put your reasoning within <think></think> tags and your final answer within <answer></answer> tags.',
     'Put only your final answer within <answer></answer> tags.'),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def verify_vendor(vendor=VENDOR):
    """Verify every preserved source before executing any of its imports."""
    vendor = Path(vendor)
    manifest = json.loads((vendor / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('Unexpected public ReasonDrive upstream commit')
    expected = manifest.get('files', {})
    for name in SOURCE_FILES:
        if not isinstance(expected.get(name), str) or sha256(vendor / name) != expected[name]:
            raise ValueError(f'Preserved upstream source hash mismatch: {name}')
    return manifest


def simple_prompt(prompt):
    """Replace only the three instructions; never interpolate a reference answer."""
    for old, new in PROMPT_REPLACEMENTS:
        if prompt.count(old) != 1:
            raise ValueError('The preserved upstream prompt no longer matches the Simple extension')
        prompt = prompt.replace(old, new, 1)
    return prompt


def derive_simple_inference(source):
    """Change only the prompt assignment inside upstream's nested format_sample."""
    tree = ast.parse(source)
    changed = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name != 'format_sample':
            continue
        for node in ast.walk(function):
            if (not isinstance(node, ast.Assign) or len(node.targets) != 1 or
                    not isinstance(node.targets[0], ast.Name) or node.targets[0].id != 'prompt'):
                continue
            if not isinstance(node.value, ast.JoinedStr):
                raise ValueError('Expected the upstream prompt to be an f-string')
            counts = [0] * len(PROMPT_REPLACEMENTS)
            for part in node.value.values:
                if not isinstance(part, ast.Constant) or not isinstance(part.value, str):
                    continue
                for index, (old, new) in enumerate(PROMPT_REPLACEMENTS):
                    counts[index] += part.value.count(old)
                    part.value = part.value.replace(old, new)
            if counts != [1, 1, 1]:
                raise ValueError('Unexpected upstream inference prompt; refusing a broad source patch')
            changed.append(node)
    if len(changed) != 1:
        raise ValueError('Expected exactly one upstream inference prompt assignment')
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + '\n'


def load_upstream(kind, mode, patches, derived_path=None, vendor=VENDOR):
    manifest = verify_vendor(vendor)
    filename = 'vlm-finetuning.py' if kind == 'train' else 'eval_finetuned_models.py'
    path = Path(vendor) / filename
    source = path.read_text(encoding='utf-8')
    if kind == 'infer' and mode == 'simple':
        source = derive_simple_inference(source)
        if derived_path is None:
            raise ValueError('The Simple inference derivative must be saved for inspection')
        Path(derived_path).write_text(source, encoding='utf-8')
        patches.append({'kind': 'simple_extension', 'target': 'inference_prompt',
                        'replacements': list(PROMPT_REPLACEMENTS),
                        'derived_source': str(derived_path), 'derived_sha256': sha256(derived_path)})
    module = types.ModuleType(f'_public_reasondrive_{kind}_{mode}')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module, manifest


def adapt_keywords(target, aliases, patches, name):
    """Translate an obsolete spelling only when the target signature proves its replacement."""
    parameters = inspect.signature(target).parameters
    renames = {old: new for old, new in aliases.items() if old not in parameters and new in parameters}

    @functools.wraps(target)
    def adapted(*args, **kwargs):
        for old, new in renames.items():
            if old not in kwargs:
                continue
            if new in kwargs:
                raise ValueError(f'Both {old} and {new} were supplied to {name}')
            kwargs[new] = kwargs.pop(old)
            record = {'kind': 'api_keyword_alias', 'target': name, 'from': old, 'to': new}
            if record not in patches:
                patches.append(record)
        return target(*args, **kwargs)
    return adapted


def install_training_compatibility(module, patches):
    module.SFTConfig = adapt_keywords(module.SFTConfig,
        {'evaluation_strategy': 'eval_strategy', 'max_seq_length': 'max_length'}, patches, 'SFTConfig')
    module.SFTTrainer = adapt_keywords(module.SFTTrainer,
        {'tokenizer': 'processing_class'}, patches, 'SFTTrainer')


def trainer_class(module, mode, patches):
    if mode == 'reason':
        return module.QwenVLTrainer
    patches.append({'kind': 'simple_extension', 'target': 'training_prompt_and_completion',
                    'replacements': list(PROMPT_REPLACEMENTS),
                    'completion': 'upstream solution.strip(); original tags retained'})

    class SimpleTrainer(module.QwenVLTrainer):
        def _format_sample(self, example):
            result = super()._format_sample(example)
            if result is None:
                return None  # Preserve the original formatter's failure behavior.
            content = result['messages'][0]['content']
            text = content[-1]
            if text.get('type') != 'text':
                raise ValueError('Unexpected upstream prompt position')
            text['text'] = simple_prompt(text['text'])
            result['messages'][1]['content'][0]['text'] = example.get('solution', '').strip()
            return result
    return SimpleTrainer


def ordered_rows(path):
    text = Path(path).read_text(encoding='utf-8')
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Inference input must be a JSON array or JSONL objects')
    if not rows:
        raise ValueError('Inference input is empty')
    return rows


@contextlib.contextmanager
def runtime_directory(directory):
    """Keep original training.log and any runtime side effects in this run directory."""
    previous_directory = Path.cwd()
    previous_mode = os.environ.get('WANDB_MODE')
    os.environ['WANDB_MODE'] = 'disabled'
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous_directory)
        if previous_mode is None:
            os.environ.pop('WANDB_MODE', None)
        else:
            os.environ['WANDB_MODE'] = previous_mode


def runtime_audit(args):
    from experiment_config import runtime_metadata
    return {'profile': 'public_baseline', 'upstream_commit': UPSTREAM_COMMIT,
            'mode': args.mode, 'command': args.command, 'model_path': args.model_path,
            'data_path': args.data_path, 'data_sha256': sha256(args.data_path),
            'image_root': args.image_root, 'wrapper_sha256': sha256(__file__),
            'environment': {'WANDB_MODE': 'disabled'}, 'runtime': runtime_metadata('cuda'),
            'compatibility_patches': [], 'status': 'running'}


def require_cuda(module):
    if not module.torch.cuda.is_available():
        raise RuntimeError('The preserved public ReasonDrive execution requires CUDA; MLX is not an upstream backend')


def collator_loss_default(collator):
    """Find the declared default through constructors accepting forwarded kwargs."""
    targets = collator.__mro__ if inspect.isclass(collator) else (collator,)
    for target in targets:
        parameters = inspect.signature(target).parameters
        parameter = parameters.get('train_on_responses_only')
        if parameter is not None:
            return parameter.default, f'{target.__module__}.{target.__qualname__}'
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            break
    return inspect.Parameter.empty, None


def latest_checkpoint(training_result):
    root = Path(training_result['actual_output_dir']).resolve()
    candidates = sorted(root.glob('checkpoint-*'),
                        key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1,
                        reverse=True)
    for path in candidates:
        required = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'training_args.bin')
        if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
            continue
        state = json.loads((path / 'trainer_state.json').read_text())
        if path.name == f"checkpoint-{state['global_step']}":
            return path
    raise ValueError(f'No complete resumable checkpoint in {root}')


def validate_resume_config(previous, current, allow_batch_change=False):
    old = previous['effective_training_config']
    if old == current:
        return
    batch_keys = {'batch_size', 'gradient_accumulation_steps'}
    if (not allow_batch_change
            or {k: v for k, v in old.items() if k not in batch_keys}
            != {k: v for k, v in current.items() if k not in batch_keys}
            or old['batch_size'] * old['gradient_accumulation_steps']
            != current['batch_size'] * current['gradient_accumulation_steps']):
        raise ValueError('Resume requires unchanged settings and the same effective batch size')


def run_train(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'training_result.json'
    resume = getattr(args, 'resume_from_checkpoint', None)
    previous = None
    if resume:
        previous = json.loads(result_path.read_text())
        resume = Path(resume).resolve()
        if previous['status'] == 'completed':
            raise ValueError('Training already completed; reuse the saved model')
        if resume != latest_checkpoint(previous):
            raise ValueError('Resume requires the latest complete checkpoint of this run')
        if (previous['mode'] != args.mode or previous['model_path'] != args.model_path
                or previous['data_sha256'] != sha256(args.data_path)
                or previous['image_root'] != args.image_root):
            raise ValueError('Resume model, mode, data or image root differs from the original run')
    elif result_path.exists():
        raise FileExistsError(f'Public run already exists; choose a fresh output directory: {output}')
    if previous:
        write_json(output / f'training_result.before_resume.{uuid.uuid4().hex}.json', previous)
    audit = runtime_audit(args)
    if resume:
        audit['actual_output_dir'] = previous['actual_output_dir']
        audit['effective_training_config'] = previous['effective_training_config']
        audit['original_training_config'] = previous.get('original_training_config', previous['effective_training_config'])
        audit['resume_from_checkpoint'] = str(resume)
        audit['resume_global_step'] = json.loads((resume / 'trainer_state.json').read_text())['global_step']
    try:
        with runtime_directory(output):
            module, manifest = load_upstream('train', args.mode, audit['compatibility_patches'])
            audit['source_manifest'] = manifest
            require_cuda(module)
            install_training_compatibility(module, audit['compatibility_patches'])
            loss_default, loss_source = collator_loss_default(module.UnslothVisionDataCollator)
            if loss_default is not False:
                raise RuntimeError('Cannot verify upstream full-text loss: installed collator must default train_on_responses_only=False')
            audit['collator_train_on_responses_only_default'] = False
            audit['collator_loss_default_source'] = loss_source
            audit['loss_policy'] = '원본 미지정; 실행 환경의 train_on_responses_only 기본 False 확인'
            audit['upstream_default_config'] = copy.deepcopy(module.DEFAULT_CONFIG)
            config = {**module.DEFAULT_CONFIG, 'model_id': args.model_path, 'data_path': args.data_path,
                      'base_image_path': args.image_root, 'output_dir': str(output),
                      'epochs': args.epochs, 'use_wandb': False,
                      'batch_size': getattr(args, 'batch_size', 1),
                      'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', 12),
                      'eval_steps': getattr(args, 'eval_steps', 100),
                      'save_steps': getattr(args, 'save_steps', 100)}
            if config['save_steps'] % config['eval_steps'] != 0:
                raise ValueError('save_steps must be a multiple of eval_steps for best-model loading')
            if (config['eval_steps'], config['save_steps']) != (10, 50):
                audit['compatibility_patches'].append({
                    'kind': 'evaluation_schedule_override', 'upstream_eval_steps': 10,
                    'upstream_save_steps': 50, 'eval_steps': config['eval_steps'],
                    'save_steps': config['save_steps']})
            if (config['batch_size'], config['gradient_accumulation_steps']) != (1, 12):
                audit['compatibility_patches'].append({
                    'kind': 'batching_override', 'upstream_batch_size': 1,
                    'upstream_gradient_accumulation_steps': 12,
                    'batch_size': config['batch_size'],
                    'gradient_accumulation_steps': config['gradient_accumulation_steps'],
                    'effective_batch_size': config['batch_size'] * config['gradient_accumulation_steps']})
            if previous:
                validate_resume_config(previous, config, getattr(args, 'allow_resume_batch_change', False))
                audit['resume_batching'] = {
                    'from_batch_size': previous['effective_training_config']['batch_size'],
                    'from_gradient_accumulation_steps': previous['effective_training_config']['gradient_accumulation_steps'],
                    'batch_size': config['batch_size'],
                    'gradient_accumulation_steps': config['gradient_accumulation_steps'],
                    'effective_batch_size': config['batch_size'] * config['gradient_accumulation_steps']}
            audit['effective_training_config'] = config
            audit['original_example_uses_qwen_branch'] = 'Qwen' in args.model_path
            trainer = trainer_class(module, args.mode, audit['compatibility_patches'])(config)
            if resume:
                trainer.output_dir = resume.parent
            audit['actual_output_dir'] = str(Path(trainer.output_dir).resolve())
            write_json(result_path, audit)
            trainer.prepare_data()  # Original QA-row 90/10 split, including its None entries.
            if resume and getattr(args, 'allow_resume_batch_change', False):
                effective_batch = config['batch_size'] * config['gradient_accumulation_steps']
                if len(trainer.train_dataset) % effective_batch:
                    raise ValueError('Batch-changing resume requires full effective batches per epoch')
            trainer.setup_trainer()  # Original 4096/2048 mismatch and step validation.
            if resume:
                if getattr(args, 'allow_resume_batch_change', False):
                    saved_state = json.loads((resume / 'trainer_state.json').read_text())
                    trainer.trainer.args.max_steps = saved_state['max_steps']
                    trainer.trainer.args.ignore_data_skip = False
                    trainer.trainer.args.auto_find_batch_size = False
                trainer.trainer.args.restore_callback_states_from_checkpoint = True
                trainer.trainer.train = functools.partial(trainer.trainer.train,
                                                         resume_from_checkpoint=str(resume))
            trainer.train()
            example = trainer.run_inference_example()  # Keep the original example/error behavior.
            audit['example_returned_none'] = example is None
            audit['status'] = 'completed'
    except BaseException as error:
        audit.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(result_path, audit)
    return audit


def run_infer(args):
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = Path(str(output) + '.runtime.json')
    if output.exists() or runtime_path.exists():
        raise FileExistsError(f'Public inference output already exists: {output}')
    audit = runtime_audit(args)
    try:
        rows = ordered_rows(args.data_path)
        audit['input_rows'] = len(rows)
        audit['inference_subset'] = 'all supplied rows in input order; no additional split'
        audit['generation'] = {'max_new_tokens': 512, 'do_sample': True, 'temperature': 0.7,
                               'top_p': 0.9, 'generation_seed': None}
        with runtime_directory(output.parent):
            module, manifest = load_upstream('infer', args.mode, audit['compatibility_patches'],
                derived_path=Path(str(output) + '.simple_derived.py'))
            audit['source_manifest'] = manifest
            require_cuda(module)
            write_json(runtime_path, audit)
            results = module.evaluate_model(args.model_path, rows, base_image_path=args.image_root,
                                            output_file=str(output))
            audit['returned_rows'] = len(results)
            audit['status'] = 'completed'
    except BaseException as error:
        audit.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(runtime_path, audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--output_dir', required=True)
    train.add_argument('--epochs', type=int, default=3)
    train.add_argument('--batch_size', type=int, default=1)
    train.add_argument('--gradient_accumulation_steps', type=int, default=12)
    train.add_argument('--eval_steps', type=int, default=100)
    train.add_argument('--save_steps', type=int, default=100)
    train.add_argument('--resume_from_checkpoint')
    train.add_argument('--allow_resume_batch_change', action='store_true')
    infer = commands.add_parser('infer')
    infer.add_argument('--output_path', required=True)
    for command in (train, infer):
        command.add_argument('--mode', choices=('simple', 'reason'), required=True)
        command.add_argument('--model_path', required=True, help='Local pinned base snapshot for train; local adapter directory for infer')
        command.add_argument('--data_path', required=True)
        command.add_argument('--image_root', required=True)
    args = parser.parse_args(argv)
    if args.command == 'train':
        if min(args.epochs, args.batch_size, args.gradient_accumulation_steps, args.eval_steps, args.save_steps) < 1:
            parser.error('training counts and step intervals must be positive')
        if args.save_steps % args.eval_steps != 0:
            parser.error('--save_steps must be a multiple of --eval_steps')
    for name in ('model_path', 'data_path', 'image_root', 'output_dir', 'output_path'):
        if hasattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    if not Path(args.model_path).is_dir():
        parser.error('--model_path must be a local model snapshot or adapter directory')
    result = run_train(args) if args.command == 'train' else run_infer(args)
    print(json.dumps({'status': result['status'], 'mode': args.mode,
                      'actual_output_dir': result.get('actual_output_dir')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
