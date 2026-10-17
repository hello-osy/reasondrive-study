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
    if row.get('status') == 'error':
        return ''
    try:
        return tagged_text(row['answer'], 'answer', required=True)
    except ValueError:
        return ''


def route_and_score(responses, data, language_backend=language_scores):
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
    counts = dict(total=len(responses), graph_excluded=0, accuracy=0, language=0, match=0, gpt_only_omitted=0)
    for entries in frames.values():
        entries.sort(key=lambda item: (item[0], item[1]))
        if entries[0][0] != 0:
            raise ValueError('Each evaluated frame needs its initial perception QA for graph filtering')
        graph = DriveLMGraph()
        for position, (_, index, qa, answer) in enumerate(entries):
            truth = qa['A']  # Official exact equality: no lowercasing or whitespace normalization.
            if position == 0:
                graph.set_graph(answer, truth)
            elif not graph.eval_graph(qa['Q']):
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


def compare(paths, data, language_backend=language_scores):
    table, details, expected, expected_meta = [], {}, None, None
    fingerprint = dataset_hash(data)
    for label, path in paths:
        if label in details:
            raise ValueError('Duplicate model label')
        rows = json.loads(Path(path).read_text())
        meta = json.loads(Path(str(path) + '.meta.json').read_text())
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
        if meta['split'] != 'test':
            raise ValueError('Only independent test results may be compared')
        if expected is not None and (truth != expected or signature != expected_meta):
            raise ValueError('Models must use identical test rows, references, and generation settings')
        expected, expected_meta = truth, signature
        scores, counts = route_and_score(rows, data, language_backend)
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
