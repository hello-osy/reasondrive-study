#!/usr/bin/env python3
"""CPU-only audit of saved ReasonDrive runs; never loads model weights or trains.

Reads JSON/source text and uses NumPy only to replay the public row splits.
All performance explanations are hypotheses, not causal findings from this audit.
"""
import argparse
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'outputs/report_experiments/public_baseline/20261027_154954_oGbbK2'
MODIFIED = ROOT / 'outputs/report_experiments/modified/20261028_180046_j4kzqit2'
PAPER_URL = 'https://arxiv.org/html/2504.10757v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    raw = Path(path).read_text(encoding='utf-8')
    return json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(line) for line in raw.splitlines() if line.strip()]


def source_record(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


def public_prompt(path, question):
    """Read the original f-string without executing the vendor module/imports."""
    tree = ast.parse(path.read_text())
    values = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'prompt' for target in node.targets)]
    if len(values) != 1 or not isinstance(values[0], ast.JoinedStr):
        raise ValueError(f'Expected one original prompt f-string in {path}')
    pieces = []
    for value in values[0].values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            pieces.append(value.value)
        elif (isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Subscript)
              and isinstance(value.value.value, ast.Name) and value.value.value.id == 'example'
              and isinstance(value.value.slice, ast.Constant) and value.value.slice.value == 'problem'):
            pieces.append(question)
        else:
            raise ValueError('Unsupported public prompt expression; audit must be reviewed')
    return ''.join(pieces)


def prompt_audit(rows, image_root, modified):
    upstream_train = ROOT / 'vendor/public_reasondrive/vlm-finetuning.py'
    upstream_infer = ROOT / 'vendor/public_reasondrive/eval_finetuned_models.py'
    current_path = ROOT / 'scripts/data_utils.py'
    spec = importlib.util.spec_from_file_location('_audit_data_utils', current_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = dict(rows[0], problem='[[QUESTION]]')
    prompts = {mode: module.format_sample(row, image_root, mode, training=False)['messages'][0]['content'][-1]['text']
               for mode in ('simple', 'reason')}
    original = public_prompt(upstream_train, row['problem'])
    original_infer = public_prompt(upstream_infer, row['problem'])
    snapshot = modified / 'source_snapshot/scripts/data_utils.py'
    return {
        'classification': 'measured_source_comparison',
        'upstream_training_reason_prompt': original,
        'upstream_inference_reason_prompt': original_infer,
        'upstream_train_inference_prompts_equal': original == original_infer,
        'current_prompts': prompts,
        'current_reason_equals_upstream': prompts['reason'] == original,
        'reason_unified_diff': '\n'.join(difflib.unified_diff(original.splitlines(), prompts['reason'].splitlines(),
                                                           fromfile='preserved_public', tofile='current')),
        'sources': [source_record(p) for p in (upstream_train, upstream_infer, current_path)],
        'saved_modified_source': source_record(snapshot) if snapshot.is_file() else None,
        'current_data_utils_matches_saved_modified_source': sha256(current_path) == sha256(snapshot) if snapshot.is_file() else None,
        'note': '현재 소스의 프롬프트 비교이며 저장된 학습 소스와의 동일성은 별도 hash로 기록한다.'}


def check_indices(indices, n):
    for name, values in indices.items():
        if any(type(i) is not int or i < 0 or i >= n for i in values) or len(values) != len(set(values)):
            raise ValueError(f'Invalid or duplicate row indices: {name}')


def overlap_audit(rows, public, modified):
    import numpy as np
    public_path = public / 'public_protocol.json'
    scene_path = modified / 'split_manifest.json'
    p, s = read_json(public_path), read_json(scene_path)
    expected_hash = content_hash(rows)
    if p['dataset_sha256'] != expected_hash or s['dataset_sha256'] != expected_hash:
        raise ValueError('Saved split dataset hash does not match current input')
    ntest = math.ceil(len(rows) * p['test_size'])
    train_order = np.random.default_rng(p['train_seed']).permutation(len(rows)).tolist()
    eval_order = np.random.default_rng(p['eval_seed']).permutation(len(rows)).tolist()
    public_sets = {'train': p['training_train_indices'], 'validation': p['training_eval_indices'], 'test': p['test_indices']}
    if public_sets != {'train': train_order[ntest:], 'validation': train_order[:ntest], 'test': eval_order[:ntest]}:
        raise ValueError('Public saved indices do not reproduce the recorded seeds')
    check_indices(public_sets, len(rows))
    check_indices(s['indices'], len(rows))
    if sorted(s['indices']['train'] + s['indices']['validation'] + s['indices']['test']) != list(range(len(rows))):
        raise ValueError('Scene manifest is not a disjoint exhaustive row partition')
    records = {}
    for name, sets, path in [('public_baseline', public_sets, public_path), ('modified', s['indices'], scene_path)]:
        train, test = set(sets['train']), set(sets['test'])
        scenes = {key: {str(rows[i]['metadata']['scene_id']) for i in ids} for key, ids in sets.items()}
        intersections = {'train_test': len(scenes['train'] & scenes['test']),
                         'train_validation': len(scenes['train'] & scenes['validation']),
                         'validation_test': len(scenes['validation'] & scenes['test'])}
        if name == 'modified' and any(intersections.values()):
            raise ValueError('Scene manifest contains scene overlap')
        records[name] = {'classification': 'measured_from_saved_manifest', 'source': source_record(path),
                         'row_counts': {key: len(ids) for key, ids in sets.items()},
                         'scene_counts': {key: len(ids) for key, ids in scenes.items()},
                         'test_rows_also_in_training': len(train & test),
                         'test_training_overlap_fraction': len(train & test) / len(test),
                         'overlap_row_indices': sorted(train & test), 'scene_intersections': intersections,
                         'validation_rows_also_in_test': len(set(sets['validation']) & test)}
    records['public_split_seeds_replayed'] = True
    records['dataset_hash_validated'] = expected_hash
    return records


def latest_trainer_state(model_dir):
    candidates = list(model_dir.glob('checkpoint-*/trainer_state.json'))
    if not candidates:
        raise FileNotFoundError(f'No trainer state under {model_dir}')
    path = max(candidates, key=lambda p: int(p.parent.name.split('-')[-1]))
    return path, read_json(path)


def training_audit(public, modified, splits):
    records = []
    for profile, run in [('public_baseline', public), ('modified', modified)]:
        for mode in ('simple', 'reason'):
            if profile == 'public_baseline':
                config_path = run / mode / 'training_result.json'
                artifact = read_json(config_path)
                config = artifact['effective_training_config']
                actual_dir = Path(artifact['actual_output_dir'])
                completed = None
            else:
                config_path = run / mode / 'training_config.json'
                config = read_json(config_path)
                artifact = {}
                actual_dir = Path(config['output_dir'])
                completed = read_json(run / mode / 'completed.json')
            state_path, state = latest_trainer_state(actual_dir)
            if completed and completed['global_step'] != state['global_step']:
                raise ValueError(f'Completed marker differs from latest state: {run / mode}')
            batch, accumulation = config['batch_size'], config['gradient_accumulation_steps']
            effective = batch * accumulation
            train_rows = splits[profile]['row_counts']['train']
            skipped_path = run / mode / 'skipped_samples.json'
            skipped = read_json(skipped_path) if skipped_path.is_file() else []
            skipped_train = sum(item['split'] == 'train' for item in skipped)
            usable_rows = train_rows - skipped_train
            steps, maximum, warmup = state['global_step'], state['max_steps'], config['warmup_steps']
            nominal_paper_updates = math.ceil(usable_rows / 12) * config['epochs']
            records.append({
                'profile': profile, 'mode': mode, 'classification': 'measured_saved_artifacts',
                'config_source': source_record(config_path), 'trainer_state_source': source_record(state_path),
                'actual_model_dir': str(actual_dir), 'microbatch': batch, 'gradient_accumulation': accumulation,
                'effective_batch_single_gpu': effective, 'train_rows_manifest': train_rows,
                'skipped_train_rows_recorded': skipped_train, 'train_rows_used_for_estimate': usable_rows,
                'optimizer_steps_completed': steps, 'optimizer_steps_planned': maximum,
                'epoch_completed': state['epoch'], 'epochs_configured': config['epochs'],
                'warmup_steps': warmup, 'warmup_fraction_of_planned_updates': warmup / maximum,
                'warmup_fraction_of_completed_updates': min(warmup, steps) / steps,
                'logged_validation_steps': [entry['step'] for entry in state['log_history'] if 'eval_loss' in entry],
                'logged_validation_losses': [entry['eval_loss'] for entry in state['log_history'] if 'eval_loss' in entry],
                'learning_rate': config['learning_rate'], 'seed': config['seed'],
                'best_model_checkpoint': state.get('best_model_checkpoint'),
                'resume_batching': artifact.get('resume_batching'),
                'response_only_configured': config.get('train_on_responses_only', artifact.get('collator_train_on_responses_only_default')),
                'counterfactual_batch12': {'classification': 'calculated_schedule_estimate_not_experiment',
                    'updates_for_same_rows_epochs': nominal_paper_updates,
                    'warmup_fraction': warmup / nominal_paper_updates,
                    'completed_updates_relative_to_estimate': steps / nominal_paper_updates,
                    'assumptions': 'single GPU; microbatch12×accum1; no early stop; ceil final batch each epoch'},
            })
    return records


def generation_audit(public, modified):
    records = []
    for profile, run in [('public_baseline', public), ('modified', modified)]:
        for mode in ('simple', 'reason'):
            path = run / f'{mode}.json.meta.json'
            metadata = read_json(path)
            records.append({'profile': profile, 'mode': mode,
                            'classification': 'measured_saved_generation_metadata',
                            'source': source_record(path),
                            'settings': {key: metadata.get(key) for key in
                                         ('do_sample', 'temperature', 'top_p', 'max_new_tokens', 'seed', 'image_size')}})
    return records


def write_report(evidence, output):
    lines = ['# ReasonDrive 기존 실험 조건 감사', '',
        '저장된 설정·trainer state·분할 인덱스·소스만 확인했다. 새 학습이나 GPU 추론을 수행한 결과가 아니다.', '',
        '## 확인된 수치', '',
        '| 실험 | 모델 | microbatch × 누적 | 유효 배치 | 완료 업데이트 | warmup 비율¹ | 검증 업데이트 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in evidence['training_runs']:
        lines.append(f"| {r['profile']} | {r['mode']} | {r['microbatch']} × {r['gradient_accumulation']} | {r['effective_batch_single_gpu']} | {r['optimizer_steps_completed']} | {r['warmup_fraction_of_planned_updates']:.2%} | {', '.join(map(str, r['logged_validation_steps']))} |")
    lines += ['', '¹ warmup_steps / trainer_state.max_steps. 공개 Reason은 500 step에서 24×1 → 8×3으로 재개되어 유효 배치24는 유지했다.', '',
              '| 분할 | train / validation / test QA | test 중 train과 같은 QA | train-test 공통 장면 |',
              '| --- | --- | --- | --- |']
    for name in ('public_baseline', 'modified'):
        r = evidence['splits'][name]
        counts = ' / '.join(str(r['row_counts'][key]) for key in ('train', 'validation', 'test'))
        lines.append(f"| {name} | {counts} | {r['test_rows_also_in_training']} / {r['row_counts']['test']} ({r['test_training_overlap_fraction']:.2%}) | {r['scene_intersections']['train_test']} |")
    lines += ['', '공개 분할은 저장된 seed3407/3047을 재실행해 확인했다. 수정본은 QA 중복 및 장면 중복이 모두 없는지 검사했다.', '',
              '## 프롬프트와 생성 조건', '',
              '공개 학습·추론 Reason 프롬프트는 동일하다. 현재 Reason 프롬프트는 문구를 줄인 별도 형식이다. 전체 문자열과 diff는 evidence.json에 저장했다.', '',
              '- 공개: 두 문장의 추가 분석 지시 + think/answer 태그 지시; sampling(temperature 0.7, top_p 0.9).',
              '- 수정본: 짧은 운전 보조 지시 + 질문 + 태그 지시; greedy decoding.', '',
              '## 논문 기준과 재현 한계', '',
              f"[논문 §4, Table 2]({PAPER_URL}#S4): Qwen2.5-VL-3B Reason Accuracy 0.47, Simple 0.32. "
              '3 epoch, 유효 배치12, LR 2e-4, warmup50, LoRA rank/alpha8, 언어 MLP 학습이다.', '',
              '논문은 정확한 분할 ID·채점 문항 수를 공개하지 않는다. GPT를 포함한 Final Score와 로컬 Final_noGPT는 다르다. '
              '따라서 현재 지표가 0.47에 도달해도 동일 조건의 논문 전체 재현으로 단정하지 않는다.', '',
              '[보존 공개 코드](https://github.com/zhourobotics/reasondrive/tree/326caecafedad1b9dbc156ebd77c8dc0eb998740), '
              '[공개 Reason adapter](https://huggingface.co/ac4462/Qwen2.5-VL-3B-DriveLM-Cot/blob/main/adapter_config.json)', '',
              '## 아직 검증하지 않은 원인 가설', '',
              '1. 배치 증가에 따른 업데이트 수 감소와 warmup 비중 변화가 Reason 학습을 약화했을 수 있다. '
              '동일 유효 배치12의 Simple/Reason 새 학습으로 확인한다.',
              '2. 프롬프트·decoding 변경이 공개 모델의 답 형식이나 행동 예측에 영향을 줬을 수 있다. '
              '같은 checkpoint와 validation 문항에서 한 요인씩 바꿔 확인한다.',
              '3. 긴 reasoning과 질문에 분산된 loss가 최종 답 학습을 약화했을 수 있다. '
              'response-only loss와 답 토큰 가중치를 각각 별도 실험한다.', '',
              '조건 선택은 validation에서 하고 고정 test는 최종 확인에 사용한다. 기존 공개 test의 학습 중복은 성능 향상의 증거로 사용하지 않는다.', '',
              '원본 파일 경로·SHA256·실제 검증 loss·counterfactual 일정 계산은 evidence.json에서 확인할 수 있다.', '']
    (output / 'audit.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/drivelm.json')
    parser.add_argument('--image-root', type=Path, default=ROOT / 'data/nuscenes')
    parser.add_argument('--public-run', type=Path, default=PUBLIC)
    parser.add_argument('--modified-run', type=Path, default=MODIFIED)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/ablation_study/audit_current')
    args = parser.parse_args()
    rows = read_rows(args.data)
    splits = overlap_audit(rows, args.public_run, args.modified_run)
    evidence = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'audit_kind': 'CPU-only saved-artifact audit; no training, inference or paid APIs',
        'script': source_record(__file__), 'data': source_record(args.data), 'row_count': len(rows),
        'prompts': prompt_audit(rows, args.image_root, args.modified_run), 'splits': splits,
        'training_runs': training_audit(args.public_run, args.modified_run, splits),
        'generation_runs': generation_audit(args.public_run, args.modified_run),
        'paper_provenance': {'url': PAPER_URL, 'version': '2504.10757v1', 'sections': ['4.1', '4.2', '4.3', 'Table 2'],
            'qwen3b_accuracy': {'simple': 0.32, 'reason': 0.47},
            'qwen3b_full_final_score': {'simple': 0.37, 'reason': 0.45},
            'training': {'epochs': 3, 'microbatch': 1, 'gradient_accumulation': 12, 'learning_rate': 0.0002,
                         'warmup_steps': 50, 'lora_r': 8, 'lora_alpha': 8, 'lora_dropout': 0,
                         'trainable': 'language MLP only', 'early_stopping_patience': 2},
            'upstream_code': 'https://github.com/zhourobotics/reasondrive/tree/326caecafedad1b9dbc156ebd77c8dc0eb998740',
            'official_adapter': 'https://huggingface.co/ac4462/Qwen2.5-VL-3B-DriveLM-Cot/blob/main/adapter_config.json'},
        'hypotheses_not_established_causes': ['Changed optimization budget/warmup fraction', 'Prompt/decoding mismatch',
                                            'Reasoning and question token loss dilution'],
        'limitations': ['Paper exact split IDs and scoring denominators are not reported.',
                       'Full paper Final Score includes GPT judging; local Final_noGPT is different.',
                       'Historical test results were already inspected; select new ablations on validation.',
                       'Checkpoint metadata establishes settings, not causal explanations for accuracy.',
                       'Counterfactual schedules are arithmetic estimates, not completed experiments.']}
    if any(name in sys.modules for name in ('torch', 'transformers', 'unsloth')):
        raise RuntimeError('CPU-only audit unexpectedly imported a model library')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_report(evidence, args.output_dir)
    print(json.dumps({'output': str(args.output_dir.resolve()), 'runs_audited': len(evidence['training_runs']),
                      'rows': len(rows), 'public_train_test_overlap': splits['public_baseline']['test_rows_also_in_training'],
                      'modified_train_test_overlap': splits['modified']['test_rows_also_in_training']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
