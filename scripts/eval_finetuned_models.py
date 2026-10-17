"""Generate responses on the independent test split saved during training."""
import argparse
import json
from pathlib import Path
from data_utils import load_split, format_sample, tagged_text, write_json, clean_message_content


def evaluate_model(model_path, rows, manifest, base_image_path, output_file,
                   mode='reason', seed=3407, max_new_tokens=512, do_sample=False):
    # Import the GPU stack only when actually running inference.
    from unsloth import FastVisionModel
    import torch
    from transformers import set_seed
    from qwen_vl_utils import process_vision_info
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model, processor = FastVisionModel.from_pretrained(model_path, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    write_json(str(output_file) + '.meta.json', {
        'dataset_sha256': manifest['dataset_sha256'], 'test_indices': manifest['indices']['test'],
        'split': 'test', 'mode': mode, 'model_path': model_path, 'seed': seed,
        'max_new_tokens': max_new_tokens, 'do_sample': do_sample,
    })
    for index in manifest['indices']['test']:
        row = rows[index]
        result = {'id': str(index), 'question': row['problem'],
                  'gt_answer': tagged_text(row.get('solution'), 'answer'),
                  'answer': '', 'final_answer': '', 'status': 'error'}
        try:
            sample = format_sample(row, base_image_path, mode, training=False)
            sample['messages'] = clean_message_content(sample['messages'])
            images, _ = process_vision_info(sample['messages'])
            text = processor.apply_chat_template(sample['messages'], add_generation_prompt=True, tokenize=False)
            inputs = processor(images=images, text=text, add_special_tokens=False, return_tensors='pt').to(device)
            # Per-row seeding prevents a skipped example from shifting later sampled outputs.
            set_seed(seed + index)
            generation = {'max_new_tokens': max_new_tokens, 'do_sample': do_sample}
            if do_sample:
                generation.update(temperature=0.7, top_p=0.9)
            with torch.inference_mode():
                output = model.generate(**inputs, **generation)
            result['answer'] = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            result['status'] = 'invalid_format'
            result['final_answer'] = tagged_text(result['answer'], 'answer', required=True)
            result['status'] = 'ok'
        except Exception as error:
            result['error'] = str(error)
            print(f'Row {index}: {error}')
        results.append(result)  # Failures remain in the denominator.
        if len(results) % 10 == 0:
            write_json(output_file, results)
    write_json(output_file, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', default='drivelm_with_thinking_800_frames_clean.json')
    parser.add_argument('--base_image_path', default='../challenge/data/nuscenes')
    parser.add_argument('--output_path', default='model_responses.json')
    parser.add_argument('--split_path', help='Defaults to split_manifest.json in model directory or its parent')
    parser.add_argument('--mode', choices=['simple', 'reason'], help='Defaults to saved training mode')
    parser.add_argument('--seed', type=int, default=3407, help='Generation seed; split comes from saved manifest')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', action='store_true', help='Opt in to seeded sampling; default is greedy')
    args = parser.parse_args()
    model_dir = Path(args.model_path)
    candidates = [model_dir, model_dir.parent]
    split_path = Path(args.split_path) if args.split_path else next(
        (p / 'split_manifest.json' for p in candidates if (p / 'split_manifest.json').is_file()), None)
    if split_path is None:
        parser.error('A saved training split manifest is required: --split_path')
    config_path = next((p / 'training_config.json' for p in candidates if (p / 'training_config.json').is_file()), None)
    config = json.loads(config_path.read_text()) if config_path else {}
    mode = args.mode or config.get('mode')
    if mode is None:
        parser.error('Specify --mode when saved training_config.json is unavailable')
    if config.get('mode') and mode != config['mode']:
        parser.error('--mode differs from the saved training mode')
    if config.get("max_steps", -1) > 0:
        parser.error("Smoke-run adapters must not be used for final evaluation")
    text = Path(args.data_path).read_text()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    saved = json.loads(split_path.read_text())
    manifest = load_split(rows, split_path, saved['seed'], saved['val_size'], saved['test_size'])
    # When a local checkpoint has its own manifest, an override must agree with it.
    own = next((p / 'split_manifest.json' for p in candidates if (p / 'split_manifest.json').is_file()), None)
    if own and json.loads(own.read_text()) != manifest:
        parser.error('Supplied split does not match this model training split')
    evaluate_model(args.model_path, rows, manifest, args.base_image_path, args.output_path,
                   mode, args.seed, args.max_new_tokens, args.do_sample)


if __name__ == '__main__':
    main()
