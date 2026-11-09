#!/usr/bin/env python3
"""CPU-only descriptive analysis of completed frozen ablation outputs.

Writes only ANALYSIS.md and analysis.json. Never changes a plan, selection,
model, response or metric artifact, and never performs model inference.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from data_utils import dataset_hash, generated_answer, tagged_text


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ('perception', 'prediction', 'planning', 'behavior')
CONTRASTS = [
    ('legacy_reason_b30_full', 'reason_b12_full', '유효 배치·업데이트 일정', True),
    ('legacy_reason_b30_full', 'reason_b30_response', 'loss 범위', True),
    ('reason_b12_full', 'reason_b12_response', 'loss 범위', False),
    ('reason_b30_response', 'reason_b12_response', '유효 배치·업데이트 일정', False),
    ('simple_b12_full', 'reason_b12_full', 'reasoning 포함 여부 / full loss', False),
    ('simple_b12_response', 'reason_b12_response', 'reasoning 포함 여부 / response loss', False),
    ('legacy_simple_b40_full', 'legacy_reason_b30_full', '기존 Simple/Reason: 배치도 다름', True),
]
PROMPT_CONTRASTS = [
    ('legacy_current_greedy', 'legacy_public_greedy', '프롬프트 / greedy'),
    ('legacy_current_sample', 'legacy_public_sample', '프롬프트 / sampling'),
    ('legacy_current_greedy', 'legacy_current_sample', 'decoding / current prompt'),
    ('legacy_public_greedy', 'legacy_public_sample', 'decoding / public prompt'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    raw = Path(path).read_text(encoding='utf-8')
    return json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(line) for line in raw.splitlines() if line.strip()]


def tag0(row):
    return 0 in row['metadata']['original_qa_pair']['tag']


def category(row):
    return row['metadata']['category']


def truth(row):
    return tagged_text(row['solution'], 'answer')


def prediction(response):
    # Matches evaluate.prediction_text; do not normalize or repair predictions.
    if response.get('status') in ('error', 'invalid_format'):
        return ''
    try:
        return generated_answer(response['answer'])
    except (KeyError, TypeError, ValueError):
        return ''


def fit_category_majority(rows, training_indices):
    """Fit only on supplied training IDs; deterministic lexical tie break."""
    if len(training_indices) != len(set(training_indices)):
        raise ValueError('Duplicate training IDs')
    counts = defaultdict(Counter)
    for index in training_indices:
        row = rows[index]
        if tag0(row):
            counts[category(row)][truth(row)] += 1
    return {name: {'answer': sorted(values, key=lambda answer: (-values[answer], answer))[0],
                   'majority_train_count': max(values.values()), 'train_tag0_count': sum(values.values()),
                   'train_unique_answer_count': len(values)}
            for name, values in counts.items()}


def scored_rows(rows, indices, responses=None, majority=None):
    if (responses is None) == (majority is None):
        raise ValueError('Exactly one prediction source is required')
    records = {}
    for index in indices:
        row = rows[index]
        if not tag0(row):
            continue
        cat, target = category(row), truth(row)
        answer = prediction(responses[str(index)]) if responses is not None else majority.get(cat, {}).get('answer', '')
        records[str(index)] = {'correct': answer == target, 'category': cat,
            'stop': cat == 'behavior' and 'the ego vehicle is not moving.' in target.lower(),
            'prediction': answer, 'truth': target, 'scene_id': str(row['metadata']['scene_id'])}
    return records


def summarize(records):
    groups = {'fixed_tag0': list(records.values()),
              **{cat: [r for r in records.values() if r['category'] == cat] for cat in CATEGORIES},
              'stop': [r for r in records.values() if r['stop']]}
    result = {}
    for name, selected in groups.items():
        n = len(selected)
        correct = sum(r['correct'] for r in selected)
        predictions = Counter(r['prediction'] for r in selected)
        truths = Counter(r['truth'] for r in selected)
        # Aggregate counts only; no raw questions, images or row-level answers in output.
        result[name] = {'correct': correct, 'n': n, 'accuracy': correct / n if n else None,
            'scene_count': len({r['scene_id'] for r in selected}),
            'unique_predictions': len(predictions), 'unique_reference_answers': len(truths),
            'most_frequent_prediction_count': max(predictions.values(), default=0),
            'most_frequent_reference_count': max(truths.values(), default=0)}
    return result


def paired_transitions(before, after):
    """Compare matched IDs, never zip files or silently drop unmatched samples."""
    if set(before) != set(after):
        raise ValueError('Paired comparison requires identical row IDs')
    if any(before[key]['truth'] != after[key]['truth'] or before[key]['category'] != after[key]['category']
           for key in before):
        raise ValueError('Paired references/categories differ')
    output = {}
    selectors = {'fixed_tag0': lambda r: True, 'behavior': lambda r: r['category'] == 'behavior',
                 'stop': lambda r: r['stop']}
    for name, include in selectors.items():
        pairs = [(before[key], after[key]) for key in before if include(before[key])]
        counts = Counter((a['correct'], b['correct']) for a, b in pairs)
        n = len(pairs)
        output[name] = {'n': n, 'both_correct': counts[True, True], 'both_wrong': counts[False, False],
            'wrong_to_correct': counts[False, True], 'correct_to_wrong': counts[True, False],
            'net_correct_change': counts[False, True] - counts[True, False],
            'accuracy_delta': (counts[False, True] - counts[True, False]) / n if n else None}
    return output


def validate_responses(path, meta, rows, expected_indices):
    responses = read_json(path)
    ids = [response.get('id') for response in responses]
    expected = {str(index) for index in expected_indices}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f'Incomplete or mismatched response IDs: {path}')
    if meta.get('dataset_sha256') != dataset_hash(rows):
        raise ValueError(f'Dataset fingerprint mismatch: {path}')
    saved_indices = meta.get('indices', meta.get('test_indices'))
    if saved_indices is None or {str(i) for i in saved_indices} != expected:
        raise ValueError(f'Metadata IDs differ from requested split: {path}')
    for response in responses:
        row = rows[int(response['id'])]
        if response.get('question') != row['problem'] or response.get('gt_answer') != truth(row):
            raise ValueError(f'Reference mismatch: {path}')
        if response.get('status') not in ('ok', 'invalid_format', 'error'):
            raise ValueError(f'Unknown response status: {path}')
        if response['status'] == 'ok' and response.get('final_answer') != generated_answer(response['answer']):
            raise ValueError(f'Parsed answer mismatch: {path}')
    return {r['id']: r for r in responses}


def check_completion_proof(path):
    proof_path = Path(str(path) + '.complete.json')
    if not proof_path.exists():
        return {'present': False, 'limitation': '원 실행에 완료 hash proof가 없음; 현재 응답·참조·분모와 저장 metrics를 교차검사함.'}
    proof = read_json(proof_path)
    paths = {'responses_sha256': path, 'metadata_sha256': Path(str(path) + '.meta.json'),
             'metrics_sha256': Path(str(path) + '.metrics.json')}
    if 'context_responses_sha256' in proof:
        paths['context_responses_sha256'] = Path(str(path) + '.context.json')
    if any(proof.get(key) != sha(value) for key, value in paths.items()):
        raise ValueError(f'Completion proof mismatch: {path}')
    return {'present': True, 'verified': True, 'source': source(proof_path)}


def training_details(model_dir):
    path = Path(model_dir) / 'training_config.json'
    if not path.is_file():
        return None
    cfg = read_json(path)
    fields = ('mode', 'epochs', 'batch_size', 'gradient_accumulation_steps', 'learning_rate', 'warmup_steps',
              'train_on_responses_only', 'max_seq_length', 'image_size', 'seed', 'evaluation_strategy',
              'lora_r', 'lora_alpha', 'finetune_attention_modules')
    return {'source': source(path), 'settings': {key: cfg.get(key) for key in fields},
            'model_source': cfg.get('model_source'), 'runtime_packages': cfg.get('runtime', {}).get('packages', {})}


def graph_entry(scores, counts):
    accuracy, n = scores.get('Accuracy'), counts.get('accuracy')
    correct = round(accuracy * n) if accuracy is not None and n is not None else None
    if correct is not None and abs(accuracy * n - correct) > 1e-6:
        raise ValueError('Stored graph Accuracy is inconsistent with its denominator')
    return {'accuracy': accuracy, 'correct': correct, 'n': n,
            'Final_noGPT': scores.get('Final_noGPT'),
            'graph_excluded_all_tags': counts.get('graph_excluded'),
            'language_n': counts.get('language'), 'match_n': counts.get('match'),
            'note': '저장된 graph metrics를 읽음; language/GPT 재채점 없음. 모델별 graph 분모가 다를 수 있음.'}


def load_completed(run, plan, rows, indices):
    records, pending = {}, []
    for variant in [*plan['variants'], *[dict(d, name='legacy_' + d['name']) for d in plan['prompt_diagnostics']]]:
        name = variant['name']
        for split in (['validation'] if name.startswith('legacy_') else ['validation', 'test']):
            path = run / 'evaluations' / f'{name}_{split}.json'
            metrics_path = Path(str(path) + '.metrics.json')
            if not metrics_path.is_file():
                pending.append({'name': name, 'split': split, 'reason': '완료 metrics 없음; 부분 응답은 분석하지 않음'})
                continue
            meta_path = Path(str(path) + '.meta.json')
            meta, stored = read_json(meta_path), read_json(metrics_path)
            selected = [i for i in indices[split] if tag0(rows[i])] if split == 'validation' else indices[split]
            responses = validate_responses(path, meta, rows, selected)
            scored = scored_rows(rows, selected, responses=responses)
            summaries = summarize(scored)
            for group in ('fixed_tag0', 'behavior', 'stop'):
                if any(stored[group][key] != summaries[group][key] for key in ('correct', 'n')):
                    raise ValueError(f'Saved fixed metrics differ from raw response exact match: {path}')
            if stored['total'] != len(responses):
                raise ValueError(f'Saved total differs: {path}')
            model = Path(plan['legacy_reason']) if name.startswith('legacy_') else run / 'models' / name
            records[(name, split)] = {'name': name, 'split': split, 'kind': 'legacy_checkpoint_diagnostic' if name.startswith('legacy_') else 'new_training',
                'fixed_metrics': summaries, 'graph_metrics': graph_entry(stored['graph_scores'], stored['graph_counts']) if split == 'test' else None,
                'total': len(responses), 'valid': sum(r['status'] == 'ok' for r in responses.values()),
                'finish_length': sum(r.get('finish_reason') == 'length' for r in responses.values()),
                'sources': [source(path), source(meta_path), source(metrics_path)], 'completion_proof': check_completion_proof(path),
                'training': training_details(model), 'generation': {key: meta.get(key) for key in ('prompt', 'do_sample', 'seed', 'budget')},
                '_scored': scored}
    for mode, name in [('simple', 'legacy_simple_b40_full'), ('reason', 'legacy_reason_b30_full')]:
        model = Path(plan['legacy_' + mode])
        root = model.parent
        path, meta_path = root / f'{mode}.json', root / f'{mode}.json.meta.json'
        comparison_path = root / 'comparison/comparison.json'
        if not path.is_file() or not comparison_path.is_file():
            pending.append({'name': name, 'split': 'test', 'reason': '기존 결과 없음'})
            continue
        meta, comparison = read_json(meta_path), read_json(comparison_path)
        responses = validate_responses(path, meta, rows, indices['test'])
        if comparison.get('provenance', {}).get(mode, {}).get('metadata') != meta:
            raise ValueError(f'Historical graph report metadata differs: {mode}')
        selected_scores = [r for r in comparison['rows'] if r['model'] == mode]
        if len(selected_scores) != 1:
            raise ValueError(f'Historical graph model missing/duplicate: {mode}')
        counts = comparison['counts'][mode]
        if counts['total'] != len(responses):
            raise ValueError('Historical graph total differs')
        scored = scored_rows(rows, indices['test'], responses=responses)
        records[(name, 'test')] = {'name': name, 'split': 'test', 'kind': 'historical_reference',
            'fixed_metrics': summarize(scored), 'graph_metrics': graph_entry(selected_scores[0], counts),
            'total': len(responses), 'valid': sum(r['status'] == 'ok' for r in responses.values()),
            'finish_length': sum(r.get('finish_reason') == 'length' for r in responses.values()),
            'sources': [source(path), source(meta_path), source(comparison_path)],
            'completion_proof': {'present': False, 'limitation': '기존 raw/metadata와 비교표를 재사용. 당시 raw hash proof는 없음.'},
            'training': training_details(model), 'generation': {'prompt': 'current', 'do_sample': meta.get('do_sample'),
                'seed': meta.get('seed'), 'budget': meta.get('max_new_tokens')}, '_scored': scored}
    return records, pending


def comparisons(records):
    result = []
    candidates = [(a, b, label, historical, 'test') for a, b, label, historical in CONTRASTS]
    candidates += [(a, b, label, False, 'validation') for a, b, label in PROMPT_CONTRASTS]
    for a, b, label, historical, split in candidates:
        before, after = records.get((a, split)), records.get((b, split))
        entry = {'before': a, 'after': b, 'split': split, 'intended_contrast': label,
                 'historical_comparator': historical, 'status': 'pending'}
        if before and after:
            entry.update(status='complete', transitions=paired_transitions(before['_scored'], after['_scored']))
            old = (before.get('training') or {}).get('settings', {})
            new = (after.get('training') or {}).get('settings', {})
            entry['recorded_setting_differences'] = {key: {'before': old.get(key), 'after': new.get(key)}
                                                   for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
            entry['interpretation_limit'] = ('과거 실행 참조이므로 실행 코드·시점 차이까지 완전히 통제한 비교는 아님.' if historical else
                '고정 계획의 같은 문항 비교. seed 한 개이며 배치 변경은 업데이트 수·warmup 비율도 바꿈.')
        result.append(entry)
    return result


def cell(item):
    if item is None or not item['n']:
        return '—'
    return f"{item['correct']}/{item['n']} ({item['accuracy']:.1%})"


def make_markdown(report):
    selected = report['selection'].get('selected_reason')
    historical = {entry['name']: entry for entry in report['results'] if entry['kind'] == 'historical_reference'}
    baseline_entry = historical.get('legacy_reason_b30_full') or historical.get('legacy_simple_b40_full')
    baseline_path = str(Path(baseline_entry['sources'][0]['path']).parent) if baseline_entry else '기존 수정본 결과 경로 확인 대기'
    lines = ['# 최종 수정본 Reason 개선 실험', '',
        '**목적: 최종 수정본에서 Simple보다 낮았던 Reason 성능을 개선하고, 어떤 변경이 영향을 주는지 확인한다.**', '',
        f'비교 기준 실행: `{baseline_path}`', '',
        '## 출발점: 최종 수정본의 기존 평가 결과', '',
        '| 모델 | graph Accuracy | Final_noGPT | 행동 예측 정답 | 정지 문항 정답 |',
        '| --- | --- | --- | --- | --- |']
    for name, label in [('legacy_simple_b40_full', 'Simple'), ('legacy_reason_b30_full', 'Reason')]:
        entry = historical.get(name)
        if entry is None:
            lines.append(f'| {label} | 확인 대기 | 확인 대기 | 확인 대기 | 확인 대기 |')
            continue
        g, m = entry['graph_metrics'], entry['fixed_metrics']
        accuracy = f"{g['accuracy']:.4f} ({g['correct']}/{g['n']})" if g and g['accuracy'] is not None else '—'
        final = f"{g['Final_noGPT']:.4f}" if g and g['Final_noGPT'] is not None else '—'
        lines.append(f"| {label} | {accuracy} | {final} | {cell(m['behavior'])} | {cell(m['stop'])} |")
    lines += ['',
        '프롬프트 진단의 `legacy_current_*`와 `legacy_public_*` 네 조건은 모두 위 **최종 수정본 Reason checkpoint**를 사용한다. '
        '`public`은 공개 코드의 프롬프트 문구를 적용한다는 뜻이며, 공개 코드로 학습한 checkpoint로 바꾼다는 뜻이 아니다.', '',
        '공개 코드 실험에서 확인한 학습·평가 데이터 중복은 별도의 과거 실행 감사 결과다. '
        '위 최종 수정본은 학습·평가 장면이 분리되어 있으므로 그 중복을 수정본 내부 Simple/Reason 성능 차이의 원인으로 설명하지 않는다.', '',
        '완료된 응답만 CPU에서 분석했다. 원 계획·모델·응답·선정 기준은 변경하지 않았다. '
        '진행 중 결과는 제외하며, test 결과로 새 조건을 고르거나 추가하지 않는다.', '',
        '## 먼저 구분해야 할 지표', '',
        '**고정 tag0 정답률 약 0.70은 논문 Accuracy 0.47과 같은 지표가 아니다.** '
        '고정 tag0는 그래프 제외 없이 모든 tag0 문항을 채점하고, graph Accuracy는 모델의 객체 식별 결과에 따라 채점 문항이 달라진다. '
        '분모와 카테고리 구성이 달라 숫자 크기를 직접 비교할 수 없다. '
        '[논문 Table 2](https://arxiv.org/html/2504.10757v1#S4)의 정확한 평가 ID도 공개되지 않았다. '
        '여기의 graph Accuracy는 DriveLM 규칙을 적용한 로컬 평가이며 Final_noGPT는 논문 Final Score와 다르다.', '',
        '## 학습셋에서만 정한 단순 기준선', '',
        '카테고리마다 학습 tag0 정답 중 가장 많은 답 하나를 항상 출력하는 기준선이다. '
        'validation/test 정답으로 답을 선택하지 않았다. 모델을 사용하지 않는다.', '',
        '| 분할 | 고정 tag0 | perception | prediction | behavior | 정지 문항 |',
        '| --- | --- | --- | --- | --- | --- |']
    for split, entry in report['majority_baselines'].items():
        m = entry['fixed_metrics']
        lines.append('| ' + split + ' | ' + ' | '.join(cell(m[k]) for k in ('fixed_tag0', 'perception', 'prediction', 'behavior', 'stop')) + ' |')
    lines += ['', '## 완료된 응답의 수치', '',
              '| 모델 / 분할 | 고정 tag0 | perception | prediction | behavior | 정지 문항 | graph Accuracy | 유효 응답 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for entry in report['results']:
        m, g = entry['fixed_metrics'], entry['graph_metrics']
        marker = ' ★검증 선정' if entry['name'] == selected else ''
        lines.append(f"| {entry['name']} / {entry['split']}{marker} | " + ' | '.join(cell(m[k]) for k in ('fixed_tag0', 'perception', 'prediction', 'behavior', 'stop'))
                     + f" | {cell(g)} | {entry['valid']}/{entry['total']} |")
    lines += ['', 'perception/prediction에서 한 가지 답만 반복해도 높은 점수를 얻는지 unique prediction 수와 학습 다수답 기준선으로 함께 확인한다. '
              '따라서 전체 고정 tag0 상승만으로 행동 판단이나 reasoning 능력 개선을 단정하지 않는다.', '',
              '## 같은 문항에서 무엇이 달라졌는가', '',
              '| 비교 (앞 → 뒤) | 범위 | 오답→정답 | 정답→오답 | 둘 다 정답 | 둘 다 오답 | 순증감 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    pending_pairs = []
    for entry in report['comparisons']:
        if entry['status'] != 'complete':
            pending_pairs.append(entry['before'] + ' → ' + entry['after'])
            continue
        label = entry['before'] + ' → ' + entry['after']
        if entry['historical_comparator']:
            label += ' (과거 참조)'
        for group in ('fixed_tag0', 'behavior', 'stop'):
            p = entry['transitions'][group]
            delta = f"{p['net_correct_change']:+d}/{p['n']}" if p['n'] else '—'
            lines.append(f"| {label} | {entry['split']} {group} | {p['wrong_to_correct']} | {p['correct_to_wrong']} | {p['both_correct']} | {p['both_wrong']} | {delta} |")
    lines += ['', '이 표는 파일 순서가 아닌 같은 QA ID끼리 짝지었다. 서로 다른 분모의 graph Accuracy 차이는 원인 효과로 계산하지 않았다. '
              '기존 batch30/full 결과는 새 batch12/full 및 batch30/response와 비교하는 참조점으로 재사용한다. '
              '과거 실행과의 비교에는 코드·환경 차이가 남을 수 있고, batch 변경은 업데이트 횟수·warmup 비중도 함께 바꾼다.', '']
    if selected:
        lines.append(f"검증 선정 기록: **{selected}**. 기존 selection.json을 읽기만 했으며 이 분석의 category/behavior 지표로 재선정하지 않았다.")
    else:
        lines.append('검증 선정은 아직 완료되지 않았다. 이 분석은 새 후보를 고르지 않는다.')
    if pending_pairs:
        lines += ['', '미완료 비교: ' + '; '.join(pending_pairs) + '.']
    lines += ['', '## 해석의 한계', '',
              '- 모든 새 후보는 사전에 고정했으나 seed는 한 개다. 표의 변화는 이 실행에서 관측한 변화이며 원인 확정이나 일반화의 증거로 확대하지 않는다.',
              '- 같은 장면에 여러 QA가 있어 서로 독립된 시행이 아니다. 단순 QA 개수로 유의성을 주장하지 않는다.',
              '- 기존 test는 이미 살펴본 자료다. 완전히 새로운 외부 평가셋으로 표현하지 않는다.',
              '- 학습 다수답 기준선과 카테고리 분석은 설명용 후속 분석이다. 사전 선정 지표인 validation fixed_tag0는 그대로 유지한다.',
              '- graph 수치는 저장된 평가 결과를 읽었으며 GPT나 language 평가를 새로 호출하지 않았다. 새 출력은 완료 proof가 있으면 확인하고, 오래된 출력의 hash proof 부재도 기록했다.', '',
              '원본 경로·SHA256·분모·고정 기준선·짝지은 변화·미완료 목록은 `analysis.json`에 있다. 원문 질문·이미지·개별 응답은 이 요약에 복사하지 않았다.', '']
    return '\n'.join(lines)


def analyze(run):
    run = Path(run).resolve()
    plan_path, split_path = run / 'plan.json', run / 'split_manifest.json'
    plan, split = read_json(plan_path), read_json(split_path)
    if sha(plan['data']) != plan['data_file_sha256'] or sha(split_path) != plan['split_sha256']:
        raise ValueError('Frozen data or split changed')
    rows, indices = read_rows(plan['data']), split['indices']
    if dataset_hash(rows) != split['dataset_sha256']:
        raise ValueError('Split fingerprint differs from dataset')
    if sorted(indices['train'] + indices['validation'] + indices['test']) != list(range(len(rows))):
        raise ValueError('Split must partition every input row exactly once')
    majority = fit_category_majority(rows, indices['train'])
    baselines = {name: {'fixed_metrics': summarize(scored_rows(rows, indices[name], majority=majority))}
                 for name in ('validation', 'test')}
    records, pending = load_completed(run, plan, rows, indices)
    pairs = comparisons(records)
    selection_path = run / 'selection.json'
    selection = read_json(selection_path) if selection_path.exists() else {'state': 'pending'}
    if selection.get('selected_reason') and selection['selected_reason'] not in {v['name'] for v in plan['variants'] if v['mode'] == 'reason'}:
        raise ValueError('Selection is not a frozen Reason candidate')
    if selection_path.exists():
        selection = {**selection, 'source': source(selection_path), 'read_only': True}
    results = []
    for key in sorted(records):
        entry = records[key]
        majority_scored = scored_rows(rows, indices[entry['split']], majority=majority)
        entry['against_train_majority'] = paired_transitions(majority_scored, entry['_scored'])
        results.append({key: value for key, value in entry.items() if not key.startswith('_')})
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_policy': 'CPU only, descriptive, aggregate, no plan/selection/model/response mutation',
        'sources': [source(__file__), source(ROOT / 'scripts/data_utils.py'), source(plan_path), source(split_path), source(plan['data'])],
        'selection': selection, 'majority_fitted_only_on': 'split.indices.train',
        'majority_fit': majority, 'majority_baselines': baselines,
        'results': results, 'comparisons': pairs, 'pending_outputs': pending,
        'limitations': ['Fixed tag0 is not paper/graph Accuracy.', 'Category/majority diagnostics are post-hoc and do not change selection.',
                       'One training seed; correlated QA within scenes; historical test already inspected.',
                       'Historical comparisons may retain code/environment differences.',
                       'Changing batch also changes update count and warmup fraction.',
                       'No claim of exact paper reproduction without the original split/evaluation contract.']}
    if any(name in sys.modules for name in ('torch', 'transformers', 'unsloth')):
        raise RuntimeError('CPU analysis unexpectedly imported a model library')
    # The only writes in this module are its own two named aggregate artifacts.
    (run / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (run / 'ANALYSIS.md').write_text(make_markdown(report), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    print(json.dumps({'run': str(args.run_dir.resolve()), 'completed_outputs_analyzed': len(result['results']),
                      'pending_outputs': len(result['pending_outputs']), 'selected_reason': result['selection'].get('selected_reason'),
                      'artifacts': ['ANALYSIS.md', 'analysis.json']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
