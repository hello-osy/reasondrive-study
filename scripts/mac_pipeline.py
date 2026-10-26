# ============================================
# 실험용 확장·설정 변경: MLX 이식과 질문 토큰 학습 정책 변경
# 논문 기반 재구현 / 공개 코드 오류 수정 부분은 아래에 구분한다.
# ============================================
"""실험용 확장·설정 변경: Apple Silicon MLX 이식 및 질문 토큰 학습 정책 변경.
논문 기반 재구현: 동일 문항의 Instruct / Simple / Reason 비교.
공개 코드 오류 수정: 독립 test/실패 행 보존. MLX optimizer는 CUDA와 다름을 기록한다.
"""
import argparse
import json
import math
import platform
from pathlib import Path
from urllib.parse import unquote, urlparse
from data_utils import dataset_hash, format_sample, load_split, tagged_text, write_json
from experiment_config import (DEFAULT_PATH, load_experiment, backend_settings, override_experiment,
                               config_hash, resolve_model_source, runtime_metadata)
from eval_finetuned_models import prompt_mode

_DEFAULT = backend_settings(load_experiment(), 'mlx')
MODEL, REVISION = _DEFAULT['model_id'], _DEFAULT['model_revision']


def read(path):
    return json.loads(Path(path).read_text())


def mac_runtime(seed=3407):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('This entry point requires Apple Silicon macOS')
    import mlx.core as mx
    import numpy as np
    mx.random.seed(seed)
    np.random.seed(seed)
    mx.set_memory_limit(10 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    return mx


def record(row, image_root, mode, training=True):
    sample = format_sample(row, image_root, mode, training)
    content = sample['messages'][0]['content']
    images = [unquote(urlparse(c['image']).path) for c in content if c['type'] == 'image']
    messages = [{'role': 'user', 'content': content[-1]['text']}]
    if training:
        messages.append({'role': 'assistant', 'content': sample['messages'][1]['content'][0]['text']})
    return {'images': images, 'messages': messages}


def open_images(paths):
    from PIL import Image
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert('RGB').resize((224, 224)))
    return images


# ============================================
# 실험용 확장·설정 변경: 이전 MLX completion-only에서 질문도 학습하도록 변경
# ============================================
def vision_dataset(records, model_config, processor):
    from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset
    class ResizedDataset(VisionDataset):
        def __getitem__(self, index):
            row = dict(self.dataset[index])
            row['images'] = open_images(row['images'])
            return self.process(row)
    # 이전 Mac completion-only에서 질문도 학습하도록 변경한다.
    # MLX는 image placeholder도 loss에 포함하므로 CUDA의 image/pad mask와 동일하지 않다.
    return ResizedDataset(records, model_config, processor, config={'train_on_completions': False})


def image_signature(rows, image_root):
    """Cheap invalidation: paths, byte sizes and nanosecond modification times."""
    import hashlib
    paths = sorted({p for row in rows for p in row['image']})
    stats = []
    for name in paths:
        path = Path(name.replace('../nuscenes', str(image_root))).resolve()
        stat = path.stat()
        stats.append((str(path), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(stats).encode()).hexdigest(), max(x[2] for x in stats)


def reusable_plan(path, expected, manifest, signature, newest_image):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        plan = read(path)
        if any(plan.get(k) != v for k, v in expected.items()):
            return None
        if plan.get('image_signature') != signature:
            # Upgrade an existing plan only when all images predate its validation.
            if 'image_signature' in plan or newest_image > path.stat().st_mtime_ns:
                return None
        for split in ('train', 'validation'):
            ids = plan['usable'][split]
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(manifest['indices'][split]):
                return None
            if any(str(i) not in plan['lengths'][mode] for mode in ('simple', 'reason') for i in ids):
                return None
        plan['image_signature'] = signature
        write_json(path, plan)
        return plan
    except (KeyError, ValueError, TypeError):
        return None


def evaluation_interval(train_size, accumulation, requested=0):
    # Align periodic validation with completed optimizer updates.
    return max(accumulation, math.ceil((requested or train_size)/accumulation)*accumulation)


def prepare(args):
    settings = backend_settings(args.experiment, 'mlx')
    mx = mac_runtime(settings['seed'])
    image_root = Path(args.image_root).expanduser().resolve()
    if not (image_root / 'samples').is_dir():
        raise ValueError('image_root must contain nuScenes samples/CAM_* image directories')
    data_path = Path(args.data_path)
    if not data_path.exists():
        from download_data import annotations
        annotations(data_path)  # 재현성: 공통 downloader의 QA_REVISION을 사용한다.
    rows = read(data_path)
    manifest = load_split(rows, args.split_path, settings['seed'], settings['val_size'], settings['test_size'], create=True)
    signature, newest_image = image_signature(rows, image_root)
    expected = {'reasoning_format_version': 3, 'data_path': str(data_path.resolve()), 'image_root': str(image_root),
                'split_path': str(Path(args.split_path).resolve()), 'dataset_sha256': dataset_hash(rows),
                'model_id': settings['model_id'], 'revision': settings['model_revision'],
                'max_seq_length': settings['max_seq_length'], 'seed': settings['seed'],
                'experiment': args.experiment, 'experiment_sha256': config_hash(args.experiment),
                'train_on_completions': False}
    if not args.rebuild and reusable_plan(args.plan, expected, manifest, signature, newest_image):
        print('Reusing validated mac_plan.json; skipped repeated image/token preprocessing')
        return
    # Only tokenizer/config files here; model weights are fetched by train.
    from huggingface_hub import snapshot_download
    from mlx_vlm.utils import load_processor
    metadata_dir = snapshot_download(settings['model_id'], revision=settings['model_revision'],
                                    allow_patterns=['*.json', '*.txt', '*.model'])
    processor = load_processor(Path(metadata_dir))
    config = read(Path(metadata_dir) / 'config.json')
    usable = {'train': [], 'validation': []}
    lengths = {'simple': {}, 'reason': {}}
    skipped = []
    from tqdm import tqdm
    for split in usable:
        for index in tqdm(manifest['indices'][split], desc=f'Check {split}'):
            try:
                for mode in ('simple', 'reason'):
                    sample = vision_dataset([record(rows[index], image_root, mode)], config, processor)[0]
                    length = int(sample['input_ids'].size)
                    if length > settings['max_seq_length']:
                        raise ValueError(f'{mode}: {length} tokens exceeds {settings["max_seq_length"]}; excluded in BOTH modes')
                    if sample.get('pixel_values') is None or length < 2:
                        raise ValueError('Missing image features or training tokens')
                    lengths[mode][str(index)] = length
                usable[split].append(index)
            except (ValueError, OSError, KeyError, TypeError) as error:
                skipped.append({'index': index, 'split': split, 'error': str(error)})
            mx.clear_cache()
    # Test must remain complete for official graph routing. Fail early on missing images.
    for index in tqdm(manifest['indices']['test'], desc='Check test images'):
        open_images(record(rows[index], image_root, 'simple', training=False)['images'])
    plan = {**expected, 'usable': usable, 'lengths': lengths, 'skipped': skipped,
            'image_signature': signature}
    write_json(args.plan, plan)
    if not all(usable.values()):
        raise ValueError('Empty train/validation after preprocessing; inspect plan skipped entries')
    print(f'Prepared paired data: {len(usable["train"])} train, {len(usable["validation"])} validation, '
          f'{len(manifest["indices"]["test"])} test; skipped {len(skipped)}. No model weights downloaded yet.')


def inputs_from_plan(args):
    plan = read(args.plan)
    if plan.get('experiment_sha256') != config_hash(args.experiment) or plan.get('train_on_completions') is not False:
        raise ValueError('Prepared plan uses different conditions/loss policy; run prepare with the same --config')
    rows = read(plan['data_path'])
    if dataset_hash(rows) != plan['dataset_sha256']:
        raise ValueError('Dataset changed; prepare a new plan')
    saved = read(plan['split_path'])
    manifest = load_split(rows, plan['split_path'], saved['seed'], saved['val_size'], saved['test_size'])
    for split, indices in plan['usable'].items():
        if not indices or len(indices) != len(set(indices)) or not set(indices) <= set(manifest['indices'][split]):
            raise ValueError('Invalid prepared training indices')
    return plan, rows, manifest


def train(args):
    settings = backend_settings(args.experiment, 'mlx')
    mx = mac_runtime(settings['seed'])
    plan, rows, manifest = inputs_from_plan(args)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output is not empty: {out}; choose a fresh output directory')
    from graph_context import load_graph_context
    context = load_graph_context(plan['split_path'], rows, manifest)
    from mlx_vlm import load
    from mlx_vlm.trainer import Trainer, VLMTrainingArgs, prepare_model_for_training, TrainingCallback
    from mlx_vlm.trainer.common.utils import save_trainable_weights
    source_path, source = resolve_model_source(plan['model_id'], plan['revision'])
    model, processor = load(source_path)
    prepare_model_for_training(model, lora_rank=settings['lora_r'], lora_alpha=settings['lora_alpha'], lora_dropout=settings['lora_dropout'],
                               target_modules=['gate_proj', 'up_proj', 'down_proj'])
    datasets = {}
    for split in ('train', 'validation'):
        indices = plan['usable'][split]
        if args.steps:
            # Exercise the longest sequence, rather than an unusually easy sample.
            indices = [max(indices, key=lambda i: plan['lengths'][args.mode][str(i)])]
        dataset = vision_dataset([record(rows[i], plan['image_root'], args.mode) for i in indices], model.config, processor)
        # Avoid reprocessing all images just to sort batches by sequence length.
        dataset.itemlen = lambda i, ids=indices: plan['lengths'][args.mode][str(ids[i])]
        datasets[split] = dataset
    iters = args.steps if args.steps else math.ceil(len(datasets['train']) * settings['epochs'])
    accumulation = min(settings['gradient_accumulation_steps'], iters)
    eval_every = evaluation_interval(len(datasets['train']), accumulation, args.eval_every)
    cfg = VLMTrainingArgs(iters=iters, batch_size=1, gradient_accumulation_steps=accumulation,
                          max_seq_length=plan['max_seq_length'], grad_checkpoint=settings['grad_checkpoint'],
                          learning_rate=settings['learning_rate'], weight_decay=settings['weight_decay'], optimizer=settings['optimizer'],
                          train_on_completions=False, cache_size=0, val_batches=-1 if not args.steps else 1,
                          steps_per_report=12, steps_per_eval=eval_every, steps_per_save=600,
                          adapter_file=str(out / 'last' / 'adapters.safetensors'),
                          clear_cache_threshold=256 * 1024**2)
    write_json(out / 'training_config.json', {**plan, 'mode': args.mode, 'iters': iters,
               'epochs_requested': settings['epochs'], 'smoke_only': bool(args.steps), 'backend': 'mlx',
               'optimizer': settings['optimizer'], 'learning_rate': settings['learning_rate'], 'warmup': False, 'scheduler': 'constant',
               'gradient_accumulation_steps': accumulation, 'train_on_completions': False,
               'grad_checkpoint': settings['grad_checkpoint'], 'eval_every': eval_every,
               'model_source': source, 'runtime': runtime_metadata('mlx')})
    write_json(out / 'split_manifest.json', manifest)
    write_json(out / 'graph_context.json', context)
    class EarlyStop(Exception):
        pass
    class BestValidation(TrainingCallback):
        def __init__(self):
            self.best, self.stale, self.history = float('inf'), 0, []
        def on_train_loss_report(self, info):
            self.history.append(info)
            write_json(out / 'history.json', self.history)
        def on_val_loss_report(self, info):
            self.history.append(info)
            write_json(out / 'history.json', self.history)
            if not math.isfinite(info['val_loss']):
                raise ValueError('Non-finite validation loss')
            if info['val_loss'] < self.best:
                self.best, self.stale = info['val_loss'], 0
                save_trainable_weights(model, str(out / 'adapters.safetensors'))
                write_json(out / 'best_validation.json', info)
            else:
                self.stale += 1
                if self.stale >= settings['early_stopping_patience']:
                    raise EarlyStop()
    callback = BestValidation()
    trainer = Trainer(model, cfg, train_dataset=datasets['train'], eval_dataset=datasets['validation'],
                      prepared=True, processing_class=processor, training_callback=callback, seed=plan['seed'])
    try:
        trainer.train()
    except EarlyStop:
        print('Early stopping; best validation adapter retained')
    write_json(out / 'completed.json', {'peak_memory_gb': mx.get_peak_memory()/1024**3,
                                       'best_val_loss': callback.best, 'smoke_only': bool(args.steps)})


# ============================================
# 논문 기반 재구현: 학습하지 않은 Instruct 기준선 추가
# 실험용 확장·설정 변경: 동일 생성 조건/출처 기록 및 graph context 별도 추론
# ============================================
def infer(args):
    settings = backend_settings(args.experiment, 'mlx')
    mx = mac_runtime(settings['seed'])
    plan, rows, manifest = inputs_from_plan(args)
    run = None if args.instruct else Path(args.model_path)
    if run is not None:
        config = read(run / 'training_config.json')
        if config['smoke_only']:
            raise ValueError('Smoke adapters cannot be used for the final comparison; run full training')
        for key in ('dataset_sha256', 'model_id', 'revision', 'image_root', 'max_seq_length', 'usable', 'lengths', 'experiment_sha256', 'train_on_completions'):
            if plan[key] != config[key]:
                raise ValueError(f'Training/plan mismatch: {key}')
        source = config.get('model_source', {})
        if source.get('resolved_revision') != plan['revision'] or source.get('processor_revision') != plan['revision'] or source.get('model_id') != plan['model_id']:
            raise ValueError('Unverified training model/processor identity')
        if read(run / 'split_manifest.json') != manifest or not (run / 'completed.json').is_file():
            raise ValueError('Missing completed run or incompatible test split')
        if read(run / 'completed.json').get('smoke_only', True):
            raise ValueError('Smoke adapter is not a final model')
        variant = config['mode']
    else:
        variant = 'instruct'
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from data_utils import generated_answer
    from graph_context import load_graph_context
    context = load_graph_context(plan['split_path'], rows, manifest)
    source_path, source = resolve_model_source(plan['model_id'], plan['revision'])
    load_options = {'adapter_path': str(run)} if run is not None else {}
    model, processor = load(source_path, **load_options)
    model.eval()
    mode = prompt_mode(variant)
    budget = settings['max_new_tokens']
    write_json(args.output_path + '.meta.json', {
        'dataset_sha256': manifest['dataset_sha256'], 'test_indices': manifest['indices']['test'],
        'split': 'test', 'mode': variant, 'variant': variant, 'prompt_mode': mode,
        'model_path': str(run) if run else plan['model_id'], 'seed': plan['seed'],
        'max_new_tokens': budget, 'do_sample': False, 'backend': 'mlx',
        'model_source': source, 'experiment_sha256': config_hash(args.experiment),
        'experiment': args.experiment, 'runtime': runtime_metadata('mlx'),
        'model_id': plan['model_id'], 'revision': plan['revision'], 'image_size': settings['image_size'],
        'graph_context_sha256': context['sha256'],
        'graph_context_path': str(Path(plan['split_path']).parent / 'graph_context.json')})

    def predict(identifier, row, row_seed, scored):
        result = {'id': str(identifier), 'question': row['problem'], 'gt_answer': tagged_text(row['solution'], 'answer'),
                  'answer': '', 'final_answer': '', 'status': 'error', 'scored': scored}
        try:
            item = record(row, plan['image_root'], mode, training=False)
            prompt = apply_chat_template(processor, model.config, item['messages'], num_images=len(item['images']))
            mx.random.seed(row_seed)
            response = generate(model, processor, prompt=prompt, image=open_images(item['images']),
                                max_tokens=budget, temperature=0.0, verbose=False)
            result['answer'] = response.text.strip()
            count = getattr(response, 'generation_tokens', None)
            result['generated_tokens'] = int(count) if count is not None else None
            result['finish_reason'] = 'length' if count is not None and count >= budget else 'not_reported_by_backend'
            result['status'] = 'invalid_format'
            result['final_answer'] = generated_answer(result['answer'])
            result['status'] = 'ok'
        except Exception as error:
            result['error'] = str(error)
        mx.clear_cache()
        return result

    results = []
    from tqdm import tqdm
    for index in tqdm(manifest['indices']['test'], desc=variant):
        results.append(predict(index, rows[index], plan['seed'] + index, True))
        if len(results) % 10 == 0:
            write_json(args.output_path, results)
    write_json(args.output_path, results)
    context_results = [predict(item['id'], item, plan['seed'] + len(rows) + i, False)
                       for i, item in enumerate(context['items'])]
    write_json(args.output_path + '.context.json', {'graph_context': context, 'responses': context_results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--image_root', required=True)
    prep.add_argument('--data_path', default='data/drivelm.json')
    prep.add_argument('--split_path', default='splits/drivelm.json')
    prep.add_argument('--max_seq_length', type=int)
    prep.add_argument('--rebuild', action='store_true', help='Ignore cached preprocessing plan')
    tr = sub.add_parser('train')
    tr.add_argument('--mode', choices=['simple', 'reason'], required=True)
    tr.add_argument('--output_dir', required=True)
    tr.add_argument('--epochs', type=int)
    tr.add_argument('--grad_checkpoint', action=argparse.BooleanOptionalAction, default=None)
    tr.add_argument('--eval_every', type=int, default=0, help='Microsteps between full validations; 0 = once per epoch')
    tr.add_argument('--steps', type=int, default=0, help='Short smoke run; not usable for final evaluation')
    ev = sub.add_parser('infer')
    ev.add_argument('--model_path')
    ev.add_argument('--instruct', action='store_true', help='Pinned base model; no adapter or training')
    ev.add_argument('--output_path', required=True)
    ev.add_argument('--max_new_tokens', type=int, help='Legacy option; must equal the shared config')
    for command in (prep, tr, ev):
        command.add_argument('--plan', default='data/mac_plan.json')
        command.add_argument('--config', default=str(DEFAULT_PATH))
    args = parser.parse_args()
    args.experiment = override_experiment(load_experiment(args.config), 'mlx',
                                         {key: getattr(args, key, None) for key in ('epochs', 'grad_checkpoint', 'max_seq_length')})
    if args.command == 'train' and (args.steps < 0 or args.eval_every < 0):
        parser.error('steps and eval_every must be nonnegative')
    if args.command == 'infer' and bool(args.model_path) == bool(args.instruct):
        parser.error('Choose either --model_path or --instruct')
    if args.command == 'infer' and args.max_new_tokens is not None and args.max_new_tokens != args.experiment['common']['max_new_tokens']:
        parser.error('--max_new_tokens must equal the shared --config value')
    {'prepare': prepare, 'train': train, 'infer': infer}[args.command](args)


if __name__ == '__main__':
    main()
