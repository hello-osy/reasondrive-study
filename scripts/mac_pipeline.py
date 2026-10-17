"""Apple Silicon MLX QLoRA: prepare, train, and generate held-out responses."""
import argparse
import json
import math
import platform
from pathlib import Path
from urllib.parse import unquote, urlparse
from data_utils import dataset_hash, format_sample, load_split, tagged_text, write_json

MODEL = 'mlx-community/Qwen2.5-VL-3B-Instruct-4bit'
REVISION = '46d4cf06a06ffc1a766c214174f9cbed2f45bcab'


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


def vision_dataset(records, model_config, processor):
    from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset
    class ResizedDataset(VisionDataset):
        def __getitem__(self, index):
            row = dict(self.dataset[index])
            row['images'] = open_images(row['images'])
            return self.process(row)
    return ResizedDataset(records, model_config, processor, config={'train_on_completions': True})


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
    mx = mac_runtime()
    image_root = Path(args.image_root).expanduser().resolve()
    if not (image_root / 'samples').is_dir():
        raise ValueError('image_root must contain nuScenes samples/CAM_* image directories')
    data_path = Path(args.data_path)
    if not data_path.exists():
        from datasets import load_dataset
        write_json(data_path, list(load_dataset('ac4462/DriveLM-reasoning', split='train')))
    rows = read(data_path)
    manifest = load_split(rows, args.split_path, create=True)
    signature, newest_image = image_signature(rows, image_root)
    expected = {'data_path': str(data_path.resolve()), 'image_root': str(image_root),
                'split_path': str(Path(args.split_path).resolve()), 'dataset_sha256': dataset_hash(rows),
                'model_id': MODEL, 'revision': REVISION, 'max_seq_length': args.max_seq_length, 'seed': 3407}
    if not args.rebuild and reusable_plan(args.plan, expected, manifest, signature, newest_image):
        print('Reusing validated mac_plan.json; skipped repeated image/token preprocessing')
        return
    # Only tokenizer/config files here; model weights are fetched by train.
    from huggingface_hub import snapshot_download
    from mlx_vlm.utils import load_processor
    metadata_dir = snapshot_download(MODEL, revision=REVISION,
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
                    if length > args.max_seq_length:
                        raise ValueError(f'{mode}: {length} tokens exceeds {args.max_seq_length}; excluded in BOTH modes')
                    if sample.get('pixel_values') is None or int(sample['completion_mask'].sum()) < 1:
                        raise ValueError('Missing image features or answer supervision')
                    lengths[mode][str(index)] = length
                usable[split].append(index)
            except (ValueError, OSError, KeyError, TypeError) as error:
                skipped.append({'index': index, 'split': split, 'error': str(error)})
            mx.clear_cache()
    # Test must remain complete for official graph routing. Fail early on missing images.
    for index in tqdm(manifest['indices']['test'], desc='Check test images'):
        open_images(record(rows[index], image_root, 'simple', training=False)['images'])
    plan = {'data_path': str(data_path.resolve()), 'image_root': str(image_root),
            'split_path': str(Path(args.split_path).resolve()), 'dataset_sha256': dataset_hash(rows),
            'model_id': MODEL, 'revision': REVISION, 'max_seq_length': args.max_seq_length,
            'usable': usable, 'lengths': lengths, 'skipped': skipped, 'seed': 3407,
            'image_signature': signature}
    write_json(args.plan, plan)
    if not all(usable.values()):
        raise ValueError('Empty train/validation after preprocessing; inspect plan skipped entries')
    print(f'Prepared paired data: {len(usable["train"])} train, {len(usable["validation"])} validation, '
          f'{len(manifest["indices"]["test"])} test; skipped {len(skipped)}. No model weights downloaded yet.')


def inputs_from_plan(args):
    plan = read(args.plan)
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
    mx = mac_runtime()
    plan, rows, manifest = inputs_from_plan(args)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output is not empty: {out}; choose a fresh output directory')
    from mlx_vlm import load
    from mlx_vlm.trainer import Trainer, VLMTrainingArgs, prepare_model_for_training, TrainingCallback
    from mlx_vlm.trainer.common.utils import save_trainable_weights
    model, processor = load(plan['model_id'], revision=plan['revision'])
    prepare_model_for_training(model, lora_rank=8, lora_alpha=8, lora_dropout=0,
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
    iters = args.steps if args.steps else math.ceil(len(datasets['train']) * args.epochs)
    accumulation = min(12, iters)
    eval_every = evaluation_interval(len(datasets['train']), accumulation, args.eval_every)
    cfg = VLMTrainingArgs(iters=iters, batch_size=1, gradient_accumulation_steps=accumulation,
                          max_seq_length=plan['max_seq_length'], grad_checkpoint=args.grad_checkpoint,
                          learning_rate=2e-4, weight_decay=0.01, optimizer='adamw',
                          train_on_completions=True, cache_size=0, val_batches=-1 if not args.steps else 1,
                          steps_per_report=12, steps_per_eval=eval_every, steps_per_save=600,
                          adapter_file=str(out / 'last' / 'adapters.safetensors'),
                          clear_cache_threshold=256 * 1024**2)
    write_json(out / 'training_config.json', {**plan, 'mode': args.mode, 'iters': iters,
               'epochs_requested': args.epochs, 'smoke_only': bool(args.steps), 'backend': 'mlx',
               'optimizer': 'adamw', 'learning_rate': 2e-4, 'warmup': False, 'scheduler': 'constant',
               'gradient_accumulation_steps': accumulation, 'train_on_completions': True,
               'grad_checkpoint': args.grad_checkpoint, 'eval_every': eval_every})
    write_json(out / 'split_manifest.json', manifest)
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
                if self.stale >= 2:
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


def infer(args):
    mx = mac_runtime()
    plan, rows, manifest = inputs_from_plan(args)
    run = Path(args.model_path)
    config = read(run / 'training_config.json')
    if config['smoke_only']:
        raise ValueError('Smoke adapters cannot be used for the final comparison; run full training')
    for key in ('dataset_sha256', 'model_id', 'revision', 'image_root', 'max_seq_length', 'usable', 'lengths'):
        if plan[key] != config[key]:
            raise ValueError(f'Training/plan mismatch: {key}')
    if read(run / 'split_manifest.json') != manifest or not (run / 'completed.json').is_file():
        raise ValueError('Missing completed run or incompatible test split')
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load(plan['model_id'], revision=plan['revision'], adapter_path=str(run))
    model.eval()
    mode = config['mode']
    write_json(args.output_path + '.meta.json', {
        'dataset_sha256': manifest['dataset_sha256'], 'test_indices': manifest['indices']['test'],
        'split': 'test', 'mode': mode, 'model_path': str(run), 'seed': plan['seed'],
        'max_new_tokens': args.max_new_tokens, 'do_sample': False, 'backend': 'mlx',
        'model_id': plan['model_id'], 'revision': plan['revision'], 'image_size': 224})
    results = []
    from tqdm import tqdm
    for index in tqdm(manifest['indices']['test'], desc=mode):
        row = rows[index]
        result = {'id': str(index), 'question': row['problem'], 'gt_answer': tagged_text(row['solution'], 'answer'),
                  'answer': '', 'final_answer': '', 'status': 'error'}
        try:
            item = record(row, plan['image_root'], mode, training=False)
            prompt = apply_chat_template(processor, model.config, item['messages'], num_images=len(item['images']))
            mx.random.seed(plan['seed'] + index)
            response = generate(model, processor, prompt=prompt, image=open_images(item['images']),
                                max_tokens=args.max_new_tokens, temperature=0.0, verbose=False)
            result['answer'] = response.text.strip()
            result['status'] = 'invalid_format'
            result['final_answer'] = tagged_text(result['answer'], 'answer', required=True)
            result['status'] = 'ok'
        except Exception as error:
            result['error'] = str(error)
        results.append(result)
        if len(results) % 10 == 0:
            write_json(args.output_path, results)
        mx.clear_cache()
    write_json(args.output_path, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--image_root', required=True)
    prep.add_argument('--data_path', default='data/drivelm.json')
    prep.add_argument('--split_path', default='splits/drivelm.json')
    prep.add_argument('--max_seq_length', type=int, default=2048)
    prep.add_argument('--rebuild', action='store_true', help='Ignore cached preprocessing plan')
    tr = sub.add_parser('train')
    tr.add_argument('--mode', choices=['simple', 'reason'], required=True)
    tr.add_argument('--output_dir', required=True)
    tr.add_argument('--epochs', type=int, default=3)
    tr.add_argument('--grad_checkpoint', action=argparse.BooleanOptionalAction, default=True)
    tr.add_argument('--eval_every', type=int, default=0, help='Microsteps between full validations; 0 = once per epoch')
    tr.add_argument('--steps', type=int, default=0, help='Short smoke run; not usable for final evaluation')
    ev = sub.add_parser('infer')
    ev.add_argument('--model_path', required=True)
    ev.add_argument('--output_path', required=True)
    ev.add_argument('--max_new_tokens', type=int, default=512)
    for command in (prep, tr, ev):
        command.add_argument('--plan', default='data/mac_plan.json')
    args = parser.parse_args()
    if args.command == 'train' and (args.epochs < 1 or args.steps < 0 or args.eval_every < 0):
        parser.error('epochs must be positive; steps must be nonnegative')
    {'prepare': prepare, 'train': train, 'infer': infer}[args.command](args)


if __name__ == '__main__':
    main()
