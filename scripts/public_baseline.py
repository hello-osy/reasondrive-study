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


def run_train(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'training_result.json'
    if result_path.exists():
        raise FileExistsError(f'Public run already exists; choose a fresh output directory: {output}')
    audit = runtime_audit(args)
    try:
        with runtime_directory(output):
            module, manifest = load_upstream('train', args.mode, audit['compatibility_patches'])
            audit['source_manifest'] = manifest
            require_cuda(module)
            install_training_compatibility(module, audit['compatibility_patches'])
            parameters = inspect.signature(module.UnslothVisionDataCollator).parameters
            loss_parameter = parameters.get('train_on_responses_only')
            if loss_parameter is None or loss_parameter.default is not False:
                raise RuntimeError('Cannot verify upstream full-text loss: installed collator must default train_on_responses_only=False')
            audit['collator_train_on_responses_only_default'] = False
            audit['loss_policy'] = '원본 미지정; 실행 환경의 train_on_responses_only 기본 False 확인'
            audit['upstream_default_config'] = copy.deepcopy(module.DEFAULT_CONFIG)
            config = {**module.DEFAULT_CONFIG, 'model_id': args.model_path, 'data_path': args.data_path,
                      'base_image_path': args.image_root, 'output_dir': str(output),
                      'epochs': args.epochs, 'use_wandb': False}
            audit['effective_training_config'] = config
            audit['original_example_uses_qwen_branch'] = 'Qwen' in args.model_path
            trainer = trainer_class(module, args.mode, audit['compatibility_patches'])(config)
            audit['actual_output_dir'] = str(Path(trainer.output_dir).resolve())
            write_json(result_path, audit)
            trainer.prepare_data()  # Original QA-row 90/10 split, including its None entries.
            trainer.setup_trainer()  # Original 4096/2048 mismatch and step validation.
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
    infer = commands.add_parser('infer')
    infer.add_argument('--output_path', required=True)
    for command in (train, infer):
        command.add_argument('--mode', choices=('simple', 'reason'), required=True)
        command.add_argument('--model_path', required=True, help='Local pinned base snapshot for train; local adapter directory for infer')
        command.add_argument('--data_path', required=True)
        command.add_argument('--image_root', required=True)
    args = parser.parse_args(argv)
    if args.command == 'train' and args.epochs < 1:
        parser.error('--epochs must be positive')
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
