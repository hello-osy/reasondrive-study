# ============================================
# 논문 기반 재구현 — 파일 전체
# 공개되지 않은 평가기를 DriveLM tag·graph·공식 언어 지표에 따라 재구현.
# ============================================
"""DriveLM tag/graph-routed metrics, with every GPT component removed."""
import argparse
import csv
import io
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from data_utils import dataset_hash, tagged_text, write_json
from vendor.drivelm_graph import DriveLMGraph
from graph_context import root_index, validate_graph_context

LANGUAGE_KEYS = ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'CIDEr')
CATEGORIES = ('perception', 'prediction', 'planning', 'behavior')


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
    except ValueError:
        return ''


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


def compare(paths, data, language_backend=language_scores):
    table, details, expected, expected_meta = [], {}, None, None
    fingerprint = dataset_hash(data)
    for label, path in paths:
        if label in details:
            raise ValueError('Duplicate model label')
        rows = json.loads(Path(path).read_text())
        meta = json.loads(Path(str(path) + '.meta.json').read_text())
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
        # Scene-disjoint test selection must include every QA from selected scenes.
        selected_scenes = {data[int(i)]['metadata']['scene_id'] for i in ids}
        required = {str(i) for i, row in enumerate(data) if row['metadata']['scene_id'] in selected_scenes}
        if set(ids) != required:
            raise ValueError('Incomplete test scenes: graph evaluation requires all QA rows')
        truth = {row['id']: (row['question'], row['gt_answer']) for row in rows}
        signature = {k: meta[k] for k in ('dataset_sha256', 'test_indices', 'split', 'seed', 'max_new_tokens', 'do_sample')}
        signature['graph_context_sha256'] = context_hash
        # ============================================
        # 실험용 확장·설정 변경: Instruct/Simple/Reason의 동일 조건 검증
        # ============================================
        # Variant and prompt_mode intentionally differ; cache paths need not be identical.
        signature['experiment_conditions'] = _experiment_conditions(meta)
        if meta['split'] != 'test':
            raise ValueError('Only independent test results may be compared')
        if expected is not None and (truth != expected or signature != expected_meta):
            raise ValueError('Models must use identical test rows, references, generation settings, '
                             'backend, image size, model/processor revisions and experiment settings')
        expected, expected_meta = truth, signature
        scores, counts = route_and_score(rows, data, language_backend, context=context)
        table.append({'model': label, **scores})
        details[label] = counts
    return table, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', nargs='+', required=True, metavar='LABEL=PATH')
    parser.add_argument('--data_path', required=True, help='Original ReasonDrive JSON/JSONL, including original_qa_pair.tag')
    parser.add_argument('--output_dir', default='results/comparison')
    args = parser.parse_args()
    pairs = [value.split('=', 1) for value in args.responses]
    if any(len(pair) != 2 for pair in pairs):
        parser.error('Use LABEL=PATH for each result file')
    table, counts = compare(pairs, read_data(args.data_path))
    output = Path(args.output_dir)
    note = ('DriveLM graph/tag routing and official language backend. No GPT calls. '
            'Restored original roots are inferred separately as graph context only; scored test rows are unchanged. '
            'Match_noGPT is coordinate F1 only (0-1). Final_noGPT=(Accuracy+Language+Match_noGPT)/3; '
            'these two columns differ from the original paper metrics. N/A means no eligible QA, not zero.')
    write_json(output / 'comparison.json', {'metric_scope': note, 'counts': counts, 'rows': table,
               'drivelm_commit': '1de72a74b257e5373400fa68239e99bd5d20580a',
               'language_evaluation_commit': '8e9e03feb7f5e605f20fc0c8dba0e24532a26216'})
    keys = list(table[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows([{k: 'N/A' if v is None else v for k, v in row.items()} for row in table])
    (output / 'comparison.csv').write_text(stream.getvalue())
    lines = [note, '', '| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join(['---'] * len(keys)) + ' |']
    for row in table:
        lines.append('| ' + ' | '.join('N/A' if row[k] is None else f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k]) for k in keys) + ' |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
