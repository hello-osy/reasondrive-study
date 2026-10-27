# ============================================
# 논문 기반 재구현 — 파일 전체
# 공개되지 않은 평가기를 DriveLM tag·graph·공식 언어 지표에 따라 재구현.
# ============================================
"""DriveLM tag/graph-routed metrics, with every GPT component removed."""
import argparse
import csv
import io
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from data_utils import dataset_hash, tagged_text, write_json
from vendor.drivelm_graph import DriveLMGraph
from graph_context import root_index, validate_graph_context

LANGUAGE_KEYS = ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'CIDEr')
CATEGORIES = ('perception', 'prediction', 'planning', 'behavior')
PROTOCOLS = ('c1_scene', 'public_baseline')
UPSTREAM_COMMIT = '326caecafedad1b9dbc156ebd77c8dc0eb998740'


def read_data(path):
    text = Path(path).read_text()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError('Expected ReasonDrive JSON array or JSONL rows')
    return rows


def final_no_gpt(accuracy, language, match):
    # ============================================
    # 실험용 확장·설정 변경: GPT 없는 별도 합성 점수
    # ============================================
    if accuracy is None or match is None or any(language[k] is None for k in LANGUAGE_KEYS):
        return None
    # Original language aggregation; remaining 0.2 weights renormalized to 1/3.
    language_score = (sum(language[k] for k in LANGUAGE_KEYS[:4])/4
                      + language['ROUGE_L'] + language['CIDEr']/10)/3
    return (accuracy + language_score + match)/3


def language_scores(predictions, references):
    java = shutil.which('java')
    if not java or subprocess.run([java, '-version'], capture_output=True).returncode:
        raise RuntimeError('Working Java runtime required for official PTB tokenization; set JAVA_HOME/PATH')
    try:
        from language_evaluation import CocoEvaluator
    except ImportError as error:
        raise RuntimeError('Install requirements-eval.txt for the official language backend') from error
    return CocoEvaluator(coco_types=['BLEU', 'ROUGE_L', 'CIDEr']).run_evaluation(predictions, references)


def prediction_text(row):
    if row.get('status') in ('error', 'invalid_format'):
        return ''
    try:
        return tagged_text(row['answer'], 'answer', required=True)
    except (KeyError, TypeError, ValueError):
        return ''


# ============================================
# 실험용 확장·설정 변경: 공개 코드 조건의 QA 분할을 명시적으로 기록·검증
# Simple 비교 모드는 확장이다. 이 protocol은 장면 독립 test를 뜻하지 않는다.
# ============================================
def public_baseline_manifest(data):
    """Reproduce Dataset.train_test_split's default shuffled row-index splits."""
    import numpy as np
    ntest = math.ceil(len(data) * 0.1)
    if not ntest or ntest >= len(data):
        raise ValueError('Public baseline needs nonempty training and evaluation subsets')
    train_order = np.random.default_rng(3407).permutation(len(data)).tolist()
    eval_order = np.random.default_rng(3047).permutation(len(data)).tolist()
    payload = {'schema_version': 1, 'protocol': 'public_baseline',
               'upstream_commit': UPSTREAM_COMMIT, 'dataset_sha256': dataset_hash(data),
               'train_seed': 3407, 'eval_seed': 3047, 'test_size': 0.1,
               'training_train_indices': train_order[ntest:],
               'training_eval_indices': train_order[:ntest],
               'test_indices': eval_order[:ntest]}
    return {**payload, 'sha256': dataset_hash(payload)}


def _public_protocol(path, meta, data):
    if meta.get('schema_version') != 2 or meta.get('protocol') != 'public_baseline':
        raise ValueError('Public baseline requires metadata schema_version=2 and protocol=public_baseline')
    if meta.get('variant') not in ('simple', 'reason'):
        raise ValueError('Public baseline supports only Simple/Reason variants')
    sidecar = Path(str(path) + '.protocol.json')
    if not sidecar.is_file():
        raise ValueError(f'Public baseline protocol manifest required: {sidecar}')
    payload = json.loads(sidecar.read_text())
    expected = public_baseline_manifest(data)
    # Recompute both partitions, rather than trusting a user-recomputed content hash.
    if dataset_hash(payload) != dataset_hash(expected) or meta.get('protocol_sha256') != expected['sha256']:
        raise ValueError('Public baseline protocol manifest differs from the pinned upstream splits')
    if meta.get('test_indices') != expected['test_indices']:
        raise ValueError('Public baseline evaluation rows/order differ from its protocol manifest')
    if (meta.get('do_sample') is not True or meta.get('max_new_tokens') != 512
            or meta.get('temperature') != 0.7 or meta.get('top_p') != 0.9):
        raise ValueError('Public baseline requires upstream sampling: 512 tokens, temperature=0.7, top_p=0.9')
    if _experiment_conditions(meta) is None:
        raise ValueError('Public baseline requires complete model/backend/experiment provenance')
    return payload


def _response_counts(responses):
    """Report raw statuses and actual strict-parser validity without dropping rows."""
    predictions = [prediction_text(row) for row in responses]
    return {
        'status_ok': sum(row.get('status') == 'ok' for row in responses),
        'status_error': sum(row.get('status') == 'error' for row in responses),
        'status_invalid_format': sum(row.get('status') == 'invalid_format' for row in responses),
        'status_unknown': sum(row.get('status') not in ('ok', 'error', 'invalid_format') for row in responses),
        'valid_answers': sum(bool(value) for value in predictions),
        'empty_predictions': sum(not value for value in predictions),
        'invalid_format': sum(not value and row.get('status') != 'error'
                              for row, value in zip(responses, predictions)),
        'finish_length': sum(row.get('finish_reason') == 'length' for row in responses),
    }


def _context_predictions(context, responses, data):
    if context is None:
        return {}
    payload = context['graph_context']
    validate_graph_context(payload, data, payload['test_indices'])
    if {str(i) for i in payload['test_indices']} != {row['id'] for row in responses}:
        raise ValueError('Graph context and scored responses use different test rows')
    expected = {item['id']: item for item in payload['items']}
    actual = context['responses']
    if len(actual) != len(expected) or {row['id'] for row in actual} != set(expected):
        raise ValueError('Context responses must include every restored root, including failures')
    result = {}
    for response in actual:
        item = expected[response['id']]
        qa = item['metadata']['original_qa_pair']
        if (response.get('scored') is not False or response['question'] != qa['Q']
                or response['gt_answer'] != qa['A']):
            raise ValueError('Context response differs from the original root annotation')
        key = (item['metadata']['scene_id'], item['metadata']['frame_id'])
        predicted = prediction_text(response)
        result[key] = (predicted, qa['A'], response.get('status') != 'ok' or not predicted)
    return result


def route_and_score(responses, data, language_backend=language_scores, context=None):
    # ============================================
    # 논문 기반 재구현: 공식 graph/tag 유지; 재구현부의 누락 root 처리 보완
    # ============================================
    # 복원 root는 별도로 추론하며 원래 채점 분모에는 추가하지 않는다.
    contexts = _context_predictions(context, responses, data)
    frames = defaultdict(list)
    for response in responses:
        index = int(response['id'])
        original = data[index]
        metadata = original.get('metadata') or {}
        qa = metadata.get('original_qa_pair') or {}
        category = metadata.get('category')
        if category not in CATEGORIES or not metadata.get('scene_id') or not metadata.get('frame_id'):
            raise ValueError(f'Row {index}: scene_id/frame_id/category metadata required')
        tags = qa.get('tag')
        if not isinstance(tags, list) or not tags or any(type(t) is not int or t not in range(4) for t in tags):
            raise ValueError(f'Row {index}: original_qa_pair.tag required; do not infer tags from question text')
        truth = tagged_text(original['solution'], 'answer')
        if qa.get('Q') != original['problem'] or str(qa.get('A', '')).strip() != truth:
            raise ValueError(f'Row {index}: original QA and reasoning dataset disagree')
        if response['question'] != original['problem'] or response['gt_answer'] != truth:
            raise ValueError(f'Row {index}: prediction reference differs from source dataset')
        frames[(metadata['scene_id'], metadata['frame_id'])].append(
            (CATEGORIES.index(category), index, qa, prediction_text(response)))
    accuracy, match, predictions, references = [], [], [], []
    counts = dict(total=len(responses), graph_excluded=0, accuracy=0, language=0, match=0,
                  gpt_only_omitted=0, context_only=len(contexts),
                  context_failures=sum(item[2] for item in contexts.values()))
    counts.update(_response_counts(responses))
    for frame_key, entries in frames.items():
        entries.sort(key=lambda item: (item[0], item[1]))
        root = root_index(data, [entry[1] for entry in entries])
        graph = DriveLMGraph()
        if root is None:
            if frame_key not in contexts:
                raise ValueError(f'Frame {frame_key} needs a separately inferred original graph root; '
                                 'run graph-context preflight and inference, without dropping test rows')
            answer, truth, _ = contexts[frame_key]
            graph.set_graph(answer, truth)
        else:
            first = next(entry for entry in entries if entry[1] == root)
            graph.set_graph(first[3], first[2]['A'])
            entries = [first] + [entry for entry in entries if entry[1] != root]
        for _, index, qa, answer in entries:
            truth = qa['A']  # Official exact equality: no lowercasing or whitespace normalization.
            if index != root and not graph.eval_graph(qa['Q']):
                counts['graph_excluded'] += 1
                continue
            tags = qa['tag']
            if 0 in tags:
                accuracy.append(float(answer == truth))
            if 2 in tags:
                predictions.append(answer)
                references.append(truth)
            if 3 in tags:
                _, score = graph.match_result(answer, truth)
                match.append(float(score))
            if set(tags) == {1}:
                counts['gpt_only_omitted'] += 1
    counts.update(accuracy=len(accuracy), language=len(predictions), match=len(match))
    language = (language_backend(predictions, references) if predictions else
                {key: None for key in LANGUAGE_KEYS})
    acc = sum(accuracy)/len(accuracy) if accuracy else None
    coordinate_match = sum(match)/len(match) if match else None
    scores = {'Accuracy': acc, **{key: language[key] for key in LANGUAGE_KEYS},
              'Match_noGPT': coordinate_match, 'Final_noGPT': final_no_gpt(acc, language, coordinate_match)}
    return scores, counts


def _experiment_conditions(meta):
    keys = ('backend', 'image_size', 'model_source', 'experiment_sha256')
    if not any(key in meta for key in keys):
        return None  # Legacy synthetic fixtures and old complete-frame result files.
    if any(key not in meta or meta[key] is None for key in keys):
        raise ValueError('Comparison requires complete backend/image/model/experiment provenance')
    source = meta['model_source']
    source_keys = ('model_id', 'resolved_revision', 'processor_revision')
    if not isinstance(source, dict) or any(not source.get(key) for key in source_keys):
        raise ValueError('Comparison requires resolved model and processor revisions')
    return {'backend': meta['backend'], 'image_size': meta['image_size'],
            'experiment_sha256': meta['experiment_sha256'],
            'model_source': {key: source[key] for key in source_keys}}


def compare(paths, data, language_backend=language_scores, protocol='c1_scene'):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown evaluation protocol: {protocol}')
    paths = list(paths)
    if not paths or (protocol == 'public_baseline' and len(paths) > 2):
        raise ValueError('Provide results; public_baseline supports at most Simple and Reason')
    table, details, expected, expected_meta = [], {}, None, None
    variants = set()
    fingerprint = dataset_hash(data)
    for label, path in paths:
        if label in details:
            raise ValueError('Duplicate model label')
        rows = json.loads(Path(path).read_text())
        meta = json.loads(Path(str(path) + '.meta.json').read_text())
        if protocol == 'public_baseline':
            public_manifest = _public_protocol(path, meta, data)
            if meta['variant'] in variants:
                raise ValueError('Duplicate public baseline model variant')
            variants.add(meta['variant'])
        elif meta.get('protocol', 'c1_scene') != 'c1_scene':
            raise ValueError('Result protocol differs from c1_scene; select --protocol explicitly')
        sidecar = Path(str(path) + '.context.json')
        context = json.loads(sidecar.read_text()) if sidecar.is_file() else None
        context_hash = context['graph_context']['sha256'] if context is not None else None
        if meta.get('graph_context_sha256') != context_hash:
            raise ValueError('Missing or mismatched graph context sidecar')
        ids = [row['id'] for row in rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Empty results or duplicate IDs')
        if set(ids) != {str(i) for i in meta['test_indices']}:
            raise ValueError('Results must contain every held-out test row, including failures')
        if meta['dataset_sha256'] != fingerprint:
            raise ValueError('Source dataset fingerprint does not match generated responses')
        if any(not str(i).isdigit() or int(i) >= len(data) for i in ids):
            raise ValueError('Invalid data row ID')
        # ============================================
        # 실험용 확장·설정 변경: 공개 코드 QA 분할만 별도 protocol으로 허용
        # C1의 완전 장면 검사와 모든 protocol의 root/context 검사는 유지한다.
        # ============================================
        if protocol == 'c1_scene':
            selected_scenes = {data[int(i)]['metadata']['scene_id'] for i in ids}
            required = {str(i) for i, row in enumerate(data) if row['metadata']['scene_id'] in selected_scenes}
            if set(ids) != required:
                raise ValueError('Incomplete test scenes: graph evaluation requires all QA rows')
        truth = {row['id']: (row['question'], row['gt_answer']) for row in rows}
        signature = {k: meta[k] for k in ('dataset_sha256', 'test_indices', 'split', 'seed', 'max_new_tokens', 'do_sample')}
        signature['graph_context_sha256'] = context_hash
        signature['protocol'] = protocol
        if protocol == 'public_baseline':
            signature['protocol_sha256'] = public_manifest['sha256']
            signature['temperature'] = meta['temperature']
            signature['top_p'] = meta['top_p']
        # ============================================
        # 실험용 확장·설정 변경: Instruct/Simple/Reason의 동일 조건 검증
        # ============================================
        # Variant and prompt_mode intentionally differ; cache paths need not be identical.
        signature['experiment_conditions'] = _experiment_conditions(meta)
        if meta['split'] != 'test':
            raise ValueError('Only the declared test/evaluation results may be compared')
        if expected is not None and (truth != expected or signature != expected_meta):
            raise ValueError('Models must use identical test rows, references, generation settings, '
                             'backend, image size, model/processor revisions and experiment settings')
        expected, expected_meta = truth, signature
        scores, counts = route_and_score(rows, data, language_backend, context=context)
        if protocol == 'public_baseline':
            train_ids = set(public_manifest['training_train_indices'])
            test_ids = set(public_manifest['test_indices'])
            train_scenes = {data[i]['metadata']['scene_id'] for i in train_ids}
            test_scenes = {data[i]['metadata']['scene_id'] for i in test_ids}
            counts.update(training_overlap_qa=len(train_ids & test_ids),
                          training_overlap_scenes=len(train_scenes & test_scenes))
        table.append({'model': label, **scores})
        details[label] = counts
    return table, details


# ============================================
# 실험용 확장·설정 변경: 같은 noGPT 채점 결과와 실패·분모를 파일/터미널에 제공
# 공개 조건과 수정 조건 사이의 차이나 개선율은 계산하지 않는다.
# ============================================
def write_report(table, counts, output_dir, protocol='c1_scene', provenance=None):
    if protocol not in PROTOCOLS or not table:
        raise ValueError('A known protocol and nonempty score table are required')
    output = Path(output_dir)
    note = ('DriveLM graph/tag routing and official language backend. No GPT calls. '
            'Restored original roots are inferred separately as graph context only; scored test rows are unchanged. '
            'Match_noGPT is coordinate F1 only (0-1). Final_noGPT=(Accuracy+Language+Match_noGPT)/3; '
            'these two columns differ from the original paper metrics. N/A means no eligible QA, not zero.')
    condition_note = ('public_baseline: upstream QA splits (train seed 3407, eval seed 3047) with a Simple-mode extension. '
                      'Training/evaluation overlap is preserved; this is not an independent scene-held-out test.'
                      if protocol == 'public_baseline' else
                      'c1_scene: all QA rows from the held-out test scenes; failures remain in the response set.')
    diagnostics = []
    fields = ('total', 'valid_answers', 'status_ok', 'status_error', 'status_invalid_format',
              'status_unknown', 'invalid_format', 'empty_predictions', 'finish_length',
              'graph_excluded', 'accuracy', 'language', 'match', 'gpt_only_omitted',
              'context_only', 'context_failures')
    if protocol == 'public_baseline':
        fields += ('training_overlap_qa', 'training_overlap_scenes')
    for row in table:
        diagnostics.append({'model': row['model'], **{key: counts[row['model']][key] for key in fields}})
    report = {'schema_version': 2, 'protocol': protocol, 'metric_scope': note,
              'condition_scope': condition_note, 'counts': counts, 'rows': table,
              'diagnostics': diagnostics, 'provenance': provenance or {},
               'drivelm_commit': '1de72a74b257e5373400fa68239e99bd5d20580a',
              'language_evaluation_commit': '8e9e03feb7f5e605f20fc0c8dba0e24532a26216'}
    write_json(output / 'comparison.json', report)
    combined = [{**row, **{f'N_{key}' if key in ('accuracy', 'language', 'match') else key: value
                           for key, value in diagnostic.items() if key != 'model'}}
                for row, diagnostic in zip(table, diagnostics)]
    keys = list(combined[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows([{k: 'N/A' if v is None else v for k, v in row.items()} for row in combined])
    (output / 'comparison.csv').write_text(stream.getvalue())
    def markdown(rows):
        keys = list(rows[0])
        def cell(value):
            return ('N/A' if value is None else f'{value:.4f}' if isinstance(value, float)
                    else str(value).replace('|', '\\|').replace('\n', ' '))
        return ['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |'] + [
            '| ' + ' | '.join(cell(row[k]) for k in keys) + ' |' for row in rows]
    def columns(names):
        return [{key: row[key] for key in names} for row in diagnostics]
    lines = [note, '', condition_note, '', 'noGPT scores', '', *markdown(table), '',
             'Response validity', '',
             'invalid_format includes strict-parser failures; empty_predictions also includes generation errors.', '',
             *markdown(columns(('model', 'total', 'valid_answers', 'status_error', 'invalid_format',
                                'empty_predictions', 'finish_length'))), '',
             'Eligible metric denominators and graph context', '',
             'Graph routing may give models different metric denominators. Context-only questions are not scored.', '',
             *markdown(columns(('model', 'accuracy', 'language', 'match', 'graph_excluded',
                                'gpt_only_omitted', 'context_only', 'context_failures'))), '',
             'Recorded response statuses', '',
             *markdown(columns(('model', 'status_ok', 'status_error', 'status_invalid_format', 'status_unknown')))]
    if protocol == 'public_baseline':
        lines += ['', 'Overlap with training data', '',
                  *markdown(columns(('model', 'training_overlap_qa', 'training_overlap_scenes')))]
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', nargs='+', required=True, metavar='LABEL=PATH')
    parser.add_argument('--data_path', required=True, help='Original ReasonDrive JSON/JSONL, including original_qa_pair.tag')
    parser.add_argument('--output_dir', default='results/comparison')
    parser.add_argument('--protocol', choices=PROTOCOLS, default='c1_scene')
    args = parser.parse_args()
    pairs = [value.split('=', 1) for value in args.responses]
    if any(len(pair) != 2 or not all(pair) for pair in pairs):
        parser.error('Use LABEL=PATH for each result file')
    table, counts = compare(pairs, read_data(args.data_path), protocol=args.protocol)
    provenance = {label: {'responses': str(Path(path).resolve()),
                          'metadata': json.loads(Path(str(path) + '.meta.json').read_text())}
                  for label, path in pairs}
    if args.protocol == 'public_baseline':
        for label, path in pairs:
            provenance[label]['protocol_manifest'] = json.loads(Path(str(path) + '.protocol.json').read_text())
    write_report(table, counts, args.output_dir, args.protocol, provenance=provenance)


if __name__ == '__main__':
    main()
