# ============================================
# 논문 기반 재구현 / 공개 코드 오류 수정 / 실험용 확장·설정 변경
# ============================================
"""논문 기반 재구현: Instruct / Simple / Reason의 공통 test 추론.
공개 코드 오류 수정: 저장 분할 재사용, 실패 행 보존.
실험용 확장·설정 변경: 고정 snapshot/생성 설정, graph context, 기존 로컬 답변 parser 보완.
"""
import argparse
import json
from pathlib import Path
from data_utils import load_split, format_sample, tagged_text, write_json, clean_message_content
from experiment_config import (DEFAULT_PATH, load_experiment, backend_settings, config_hash,
                               resolve_model_source, runtime_metadata)


# ============================================
# 논문 기반 재구현: Instruct는 학습 없이 Simple과 동일한 answer-only 입력 사용
# ============================================
def prompt_mode(variant):
    if variant not in ('instruct', 'simple', 'reason'):
        raise ValueError('Unknown model variant')
    return 'reason' if variant == 'reason' else 'simple'


# ============================================
# 실험용 확장·설정 변경: 실험 설정과 학습 artifact 출처 검증
# ============================================
def validate_training_run(config, manifest, model_dir, experiment):
    """학습 완료·snapshot·설정이 확인된 adapter만 최종 비교에 사용한다."""
    if not config or config.get('max_steps', -1) > 0:
        raise ValueError('A completed full-training configuration is required; smoke adapters are rejected')
    if config.get('experiment_sha256') != config_hash(experiment):
        raise ValueError('Training/inference experiment settings differ')
    source = config.get('model_source', {})
    expected = backend_settings(experiment, 'cuda')
    if (source.get('model_id') != expected['model_id'] or
            source.get('resolved_revision') != expected['model_revision'] or
            source.get('processor_revision') != expected['model_revision']):
        raise ValueError('Training model/processor identity is missing or differs from the pinned experiment')
    if json.loads((model_dir / 'split_manifest.json').read_text()) != manifest:
        raise ValueError('Supplied split differs from training split')
    completed = json.loads((model_dir / 'completed.json').read_text())
    if completed.get('smoke', True):
        raise ValueError('Smoke-run adapters must not be used for final evaluation')


# ============================================
# 공개 코드 오류 수정: 실패한 scored row도 결과에 보존
# 논문 기반 재구현: 원본에 실행 연결이 없던 Instruct 기준선
# 실험용 확장·설정 변경: 생성 종료 정보·graph context sidecar·로컬 parser 보완
# ============================================
def evaluate_model(model_path, rows, manifest, base_image_path, output_file,
                   variant, experiment, context, context_path):
    # GPU/model imports occur only when the user runs inference.
    from unsloth import FastVisionModel
    import torch
    from transformers import set_seed
    from qwen_vl_utils import process_vision_info
    from data_utils import generated_answer
    settings = backend_settings(experiment, 'cuda')
    seed, budget = settings['seed'], settings['max_new_tokens']
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA inference requires a CUDA GPU')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_path, source = resolve_model_source(settings['model_id'], settings['model_revision'])
    model, processor = FastVisionModel.from_pretrained(source_path, load_in_4bit=True)
    if variant != 'instruct':
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    FastVisionModel.for_inference(model)
    mode = prompt_mode(variant)
    write_json(str(output_file) + '.meta.json', {
        'dataset_sha256': manifest['dataset_sha256'], 'test_indices': manifest['indices']['test'],
        'split': 'test', 'mode': variant, 'variant': variant, 'prompt_mode': mode,
        'model_path': model_path if variant != 'instruct' else settings['model_id'], 'seed': seed,
        'max_new_tokens': budget, 'do_sample': False, 'backend': 'cuda',
        'model_source': source, 'experiment_sha256': config_hash(experiment),
        'experiment': experiment, 'runtime': runtime_metadata('cuda'),
        'image_size': settings['image_size'], 'graph_context_sha256': context['sha256'],
        'graph_context_path': str(Path(context_path).resolve())})

    def predict(identifier, row, row_seed, scored):
        result = {'id': str(identifier), 'question': row['problem'],
                  'gt_answer': tagged_text(row.get('solution'), 'answer'),
                  'answer': '', 'final_answer': '', 'status': 'error', 'scored': scored}
        try:
            sample = format_sample(row, base_image_path, mode, training=False)
            sample['messages'] = clean_message_content(sample['messages'])
            images, _ = process_vision_info(sample['messages'])
            text = processor.apply_chat_template(sample['messages'], add_generation_prompt=True, tokenize=False)
            inputs = processor(images=images, text=text, add_special_tokens=False, return_tensors='pt').to('cuda')
            set_seed(row_seed)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=budget, do_sample=False)
            generated = output[:, inputs.input_ids.shape[1]:]
            result['generated_tokens'] = int(generated.shape[1])
            eos = processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else processor.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            ended = bool(generated.shape[1]) and int(generated[0, -1].item()) in eos
            result['finish_reason'] = 'eos' if ended else ('length' if generated.shape[1] >= budget else 'other')
            result['answer'] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            result['status'] = 'invalid_format'
            result['final_answer'] = generated_answer(result['answer'])
            result['status'] = 'ok'
        except Exception as error:
            result['error'] = str(error)
            print(f'Row {identifier}: {error}')
        return result

    results = []
    for index in manifest['indices']['test']:
        results.append(predict(index, rows[index], seed + index, True))
        if len(results) % 10 == 0:
            write_json(output_file, results)
    write_json(output_file, results)
    context_results = [predict(item['id'], item, seed + len(rows) + i, False)
                       for i, item in enumerate(context['items'])]
    write_json(str(output_file) + '.context.json', {'graph_context': context, 'responses': context_results})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_PATH))
    parser.add_argument('--model_path', help='Completed adapter directory; omitted for --instruct')
    parser.add_argument('--instruct', action='store_true', help='Evaluate the pinned base model without training/adapters')
    parser.add_argument('--data_path', default='data/drivelm.json')
    parser.add_argument('--base_image_path', default='data/nuscenes')
    parser.add_argument('--output_path', default='model_responses.json')
    parser.add_argument('--split_path', help='Saved split; mandatory for --instruct')
    parser.add_argument('--mode', choices=['simple', 'reason'], help='Must match saved training mode')
    parser.add_argument('--seed', type=int, help='Legacy option; must equal the shared config')
    parser.add_argument('--max_new_tokens', type=int, help='Legacy option; must equal the shared config')
    parser.add_argument('--do_sample', action='store_true', help='Rejected by the fixed greedy C1 protocol')
    args = parser.parse_args()
    experiment = load_experiment(args.config)
    settings = backend_settings(experiment, 'cuda')
    for key in ('seed', 'max_new_tokens'):
        if getattr(args, key) is not None and getattr(args, key) != settings[key]:
            parser.error(f'--{key} must equal the shared --config value')
    if args.do_sample:
        parser.error('The shared C1 protocol uses greedy generation')
    if args.instruct and (args.model_path or args.mode):
        parser.error('--instruct uses the pinned base and answer-only prompt; omit --model_path/--mode')
    if not args.instruct and not args.model_path:
        parser.error('--model_path or --instruct is required')
    candidates = [] if args.instruct else [Path(args.model_path), Path(args.model_path).parent]
    model_dir = next((p for p in candidates if (p / 'training_config.json').is_file()), None)
    split_path = Path(args.split_path) if args.split_path else (model_dir / 'split_manifest.json' if model_dir else None)
    if split_path is None:
        parser.error('A saved training split manifest is required: --split_path')
    text = Path(args.data_path).read_text()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    saved = json.loads(split_path.read_text())
    manifest = load_split(rows, split_path, saved['seed'], saved['val_size'], saved['test_size'])
    if args.instruct:
        variant = 'instruct'
    else:
        if model_dir is None:
            parser.error('Saved training_config.json and completed.json are required')
        config = json.loads((model_dir / 'training_config.json').read_text())
        validate_training_run(config, manifest, model_dir, experiment)
        variant = config['mode']
        if args.mode and args.mode != variant:
            parser.error('--mode differs from the saved training mode')
    from graph_context import load_graph_context
    context = load_graph_context(split_path, rows, manifest)
    evaluate_model(args.model_path, rows, manifest, args.base_image_path, args.output_path,
                   variant, experiment, context, split_path.parent / 'graph_context.json')


if __name__ == '__main__':
    main()
