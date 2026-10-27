# ============================================
# 실험용 확장·설정 변경 — 파일 전체
# 공개 코드 조건과 수정 조건을 각각 실행·기록하고 공통 noGPT 결과표로 정리한다.
# 원본의 학습·평가 문제를 고치는 경로와 비교용으로 보존하는 경로를 분리한다.
# ============================================
"""Report experiment orchestration. CLI paths are relative to the project root."""
import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from data_utils import dataset_hash, generated_answer, tagged_text, write_json

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('public_baseline', 'modified')
TITLES = {'public_baseline': '실험 1 · 공개 코드 기반 (Simple 모드 확장)',
          'modified': '실험 2 · 수정본'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(run_dir, **values):
    path = Path(run_dir) / 'status.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(values, updated_at=datetime.now(timezone.utc).isoformat())
    write_json(path, previous)


def start(run_dir, profile):
    """Save the actual source, including uncommitted changes, before running."""
    run_dir = Path(run_dir)
    inventory = {}
    candidates = [*ROOT.glob('scripts/**/*.py'), *ROOT.glob('scripts/*.sh'),
                  *ROOT.glob('configs/*.json'), *ROOT.glob('requirements*.txt'),
                  *ROOT.glob('vendor/public_reasondrive/*')]
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        target = run_dir / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        inventory[str(relative)] = digest(target)
    write_json(run_dir / 'source_manifest.json', inventory)
    state(run_dir, schema_version=1, profile=profile, status='running', stage='environment',
          source_sha256=dataset_hash(inventory))


def run_step(run_dir, stage, arguments):
    state(run_dir, stage=stage)
    command = [sys.executable, *map(str, arguments)]
    print('+ ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True,
                   env={**os.environ, 'WANDB_MODE': 'disabled', 'WANDB_DISABLED': 'true'})


def normalize_public(raw, records):
    """Recover skipped rows without changing raw model answers or sampling order.

    records are (standard_id, source_row, scored). The upstream ID includes the
    enumeration position, so duplicate questions cannot silently misalign rows.
    """
    expected = {}
    for position, (identifier, row, scored) in enumerate(records):
        metadata = row['metadata']
        raw_id = f"{metadata['scene_id']}_{metadata['frame_id']}_{position}"
        expected[raw_id] = (str(identifier), row, scored)
    received = {}
    for item in raw:
        key = item['id']
        if key not in expected or key in received:
            raise ValueError('Unexpected or duplicate upstream response ID')
        _, original, _ = expected[key]
        truth = tagged_text(original['solution'], 'answer')
        if item['question'] != '<image>\n' + original['problem'] or item['gt_answer'] != truth:
            raise ValueError('Upstream response does not match the recorded source row')
        if not isinstance(item.get('answer'), str):
            raise ValueError('Upstream answer must be raw text')
        received[key] = item
    result = []
    for raw_id, (identifier, original, scored) in expected.items():
        row = {'id': identifier, 'question': original['problem'],
               'gt_answer': tagged_text(original['solution'], 'answer'),
               'answer': '', 'final_answer': '', 'status': 'error', 'scored': scored,
               'upstream_id': raw_id, 'finish_reason': 'unavailable_in_upstream'}
        if raw_id in received:
            row['answer'] = received[raw_id]['answer']
            try:
                row['final_answer'] = generated_answer(row['answer'])
                row['status'] = 'ok'
            except ValueError as error:
                row.update(status='invalid_format', error=str(error))
        else:
            row['error'] = 'Upstream skipped this row; see raw inference log'
        result.append(row)
    return result


def public_settings(experiment):
    """Only model snapshot and epochs are configurable for the preserved baseline."""
    from experiment_config import backend_settings
    settings = backend_settings(experiment, 'cuda')
    fixed = {'batch_size': 1, 'gradient_accumulation_steps': 12, 'learning_rate': 2e-4,
             'weight_decay': .01, 'warmup_steps': 50, 'lora_r': 8, 'lora_alpha': 8,
             'lora_dropout': 0, 'seed': 3407, 'test_size': .1, 'max_seq_length': 2048,
             'image_size': 224, 'max_new_tokens': 512}
    if any(settings[key] != value for key, value in fixed.items()):
        raise ValueError('public_baseline preserves upstream defaults; only model/revision/epochs overrides are supported')
    return {**fixed, 'epochs': settings['epochs'], 'model_id': settings['model_id'],
            'model_revision': settings['model_revision'], 'train_seed': 3407,
            'eval_split_seed': 3047, 'generation_seed': None, 'collator_max_seq_length': 4096,
            'evaluation_strategy': 'steps', 'eval_steps': 10, 'save_steps': 50,
            'early_stopping_patience': 2, 'do_sample': True, 'temperature': .7, 'top_p': .9,
            'loss_policy': 'upstream_unspecified_runtime_default_false',
            'split_policy': 'QA 90/10; final evaluation resplit with seed 3047',
            'extensions': ['Simple answer-only mode', 'runtime keyword compatibility',
                           'fixed base-model snapshot and 3B CLI selection',
                           'common noGPT scoring, context-only roots and failed-row accounting']}


def run_public(args):
    from evaluate import public_baseline_manifest, read_data, compare, write_report
    from experiment_config import load_experiment, resolve_model_source, runtime_metadata
    from graph_context import prepare_graph_context
    run_dir = Path(args.run_dir).resolve()
    data_path, image_root = Path(args.data_path).resolve(), Path(args.image_root).resolve()
    experiment = load_experiment(args.config)
    effective = public_settings(experiment)
    data = read_data(data_path)
    protocol = public_baseline_manifest(data)
    write_json(run_dir / 'public_protocol.json', protocol)
    write_json(run_dir / 'public_settings.json', effective)
    manifest = {'dataset_sha256': dataset_hash(data), 'indices': {'test': protocol['test_indices']}}
    state(run_dir, stage='graph_context')
    context = prepare_graph_context(data, manifest, run_dir / 'public_split.json')
    state(run_dir, stage='base_model')
    snapshot, source = resolve_model_source(effective['model_id'], effective['model_revision'])
    write_json(run_dir / 'base_model.json', source)
    # Both train from the base snapshot in independent processes, never from each other's adapter.
    models = {}
    for mode in ('simple', 'reason'):
        output = run_dir / mode
        run_step(run_dir, 'train_' + mode, [ROOT / 'scripts/public_baseline.py', 'train',
                 '--mode', mode, '--model_path', snapshot, '--data_path', data_path,
                 '--image_root', image_root, '--output_dir', output, '--epochs', effective['epochs']])
        trained = json.loads((output / 'training_result.json').read_text())
        if trained['status'] != 'completed':
            raise ValueError('Upstream training did not complete')
        models[mode] = trained['actual_output_dir']
    records = [(str(i), data[i], True) for i in protocol['test_indices']]
    records += [(item['id'], item, False) for item in context['items']]
    ordered_path = run_dir / 'evaluation_inputs.json'
    write_json(ordered_path, [row for _, row, _ in records])
    write_json(run_dir / 'evaluation_order.json', [{'id': i, 'scored': s} for i, _, s in records])
    paths = []
    for mode in ('simple', 'reason'):
        raw_path = run_dir / f'{mode}.raw.json'
        run_step(run_dir, 'infer_' + mode, [ROOT / 'scripts/public_baseline.py', 'infer',
                 '--mode', mode, '--model_path', models[mode], '--data_path', ordered_path,
                 '--image_root', image_root, '--output_path', raw_path])
        responses = normalize_public(json.loads(raw_path.read_text()), records)
        path = run_dir / f'{mode}.json'
        write_json(path, [row for row in responses if row['scored']])
        write_json(str(path) + '.context.json', {'graph_context': context,
                   'responses': [row for row in responses if not row['scored']]})
        write_json(str(path) + '.protocol.json', protocol)
        write_json(str(path) + '.meta.json', {
            'schema_version': 2, 'protocol': 'public_baseline', 'protocol_sha256': protocol['sha256'],
            'dataset_sha256': dataset_hash(data), 'test_indices': protocol['test_indices'],
            'split': 'test', 'seed': None, 'variant': mode, 'mode': mode, 'prompt_mode': mode,
            'max_new_tokens': 512, 'do_sample': True, 'temperature': .7, 'top_p': .9,
            'backend': 'cuda', 'image_size': 224, 'model_source': source,
            'experiment_sha256': dataset_hash(effective), 'effective_settings': effective,
            'graph_context_sha256': context['sha256'], 'runtime': runtime_metadata('cuda'),
            'source_sha256': json.loads((run_dir / 'status.json').read_text())['source_sha256']})
        paths.append((mode, path))
    state(run_dir, stage='score_no_gpt')
    table, counts = compare(paths, data, protocol='public_baseline')
    write_report(table, counts, run_dir / 'comparison', protocol='public_baseline',
                 provenance={'settings': effective, 'protocol_manifest': protocol, 'base_model': source})
    complete(run_dir, 'public_baseline')


def complete(run_dir, profile):
    run_dir = Path(run_dir).resolve()
    report_path = run_dir / 'comparison/comparison.json'
    report = json.loads(report_path.read_text())
    expected_protocol = 'public_baseline' if profile == 'public_baseline' else 'c1_scene'
    if report['protocol'] != expected_protocol or {r['model'] for r in report['rows']} != {'simple', 'reason'}:
        raise ValueError('Completion requires both Simple and Reason from the expected protocol')
    inputs = [report_path, report_path.with_suffix('.md'), report_path.with_suffix('.csv'),
              run_dir / 'experiment.json', run_dir / 'source_manifest.json']
    for mode in ('simple', 'reason'):
        inputs += [run_dir / f'{mode}.json', run_dir / f'{mode}.json.meta.json',
                   run_dir / f'{mode}.json.context.json']
        if profile == 'public_baseline':
            inputs += [run_dir / f'{mode}.raw.json', run_dir / f'{mode}.json.protocol.json',
                       run_dir / mode / 'training_result.json']
        else:
            inputs += [run_dir / mode / 'completed.json', run_dir / mode / 'training_config.json']
    files = {str(p.relative_to(run_dir)): digest(p) for p in inputs}
    record = {'schema_version': 1, 'profile': profile, 'protocol': expected_protocol,
              'status': 'completed', 'files': files, 'comparison_type': 'descriptive_unpaired'}
    write_json(run_dir / 'run.json', record)
    state(run_dir, status='completed', stage='finished')
    write_json(run_dir.parent / 'latest.json', {'schema_version': 1, 'profile': profile,
               'run_dir': run_dir.name, 'run_sha256': digest(run_dir / 'run.json')})
    print(f'완료: {report_path.parent / "comparison.md"}')


def load_completed(root, profile):
    parent = Path(root) / profile
    latest_path = parent / 'latest.json'
    if not latest_path.exists():
        return None
    latest = json.loads(latest_path.read_text())
    name = latest['run_dir']
    if latest.get('profile') != profile or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Invalid latest run pointer')
    run = parent / name
    if digest(run / 'run.json') != latest['run_sha256']:
        raise ValueError('Completed run manifest changed')
    record = json.loads((run / 'run.json').read_text())
    if record.get('status') != 'completed' or record.get('profile') != profile:
        raise ValueError('Incomplete/mismatched experiment')
    for name, fingerprint in record['files'].items():
        target = run / name
        if not target.resolve().is_relative_to(run.resolve()) or digest(target) != fingerprint:
            raise ValueError(f'Completed artifact changed: {name}')
    return run, json.loads((run / 'comparison/comparison.json').read_text())


def summarize(root):
    root = Path(root)
    entries, rows = [], []
    for profile in PROFILES:
        completed = load_completed(root, profile)
        if completed is None:
            continue
        run, report = completed
        entries.append({'profile': profile, 'run_dir': str(run.relative_to(root)), 'report': report})
        for item in report['rows']:
            count = report['counts'][item['model']]
            rows.append({'experiment': profile, 'protocol': report['protocol'], **item,
                         'N': count['total'], 'valid_answers': count['valid_answers'],
                         'generation_errors': count['status_error'], 'invalid_format': count['invalid_format'],
                         'N_accuracy': count['accuracy'], 'N_language': count['language'],
                         'N_match': count['match'], 'graph_excluded': count['graph_excluded'],
                         'training_overlap_qa': count.get('training_overlap_qa', 0)})
    if not rows:
        raise ValueError('완료된 실험이 없습니다. 학습·추론·채점이 모두 끝난 후 다시 실행하세요.')
    note = ('두 실험은 같은 noGPT 평가기를 사용하지만 학습·평가 문항, 모델 선택과 생성 조건이 다르다. '
            '각 실험의 결과를 별도로 제시하며 점수 차이를 코드 수정만의 효과로 해석하지 않는다. '
            'Match_noGPT와 Final_noGPT는 논문의 GPT 포함 점수와 다른 지표다. N/A는 대상 없음이다.')
    write_json(root / 'comparison.json', {'schema_version': 1, 'comparison_type': 'descriptive_unpaired',
               'metric_scope': note, 'experiments': entries, 'rows': rows})
    stream = io.StringIO();writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows([{k: 'N/A' if v is None else v for k, v in r.items()} for r in rows])
    (root / 'comparison.csv').write_text(stream.getvalue())
    lines = ['# 보고서용 noGPT 실험 결과', '', note, '']
    for profile in PROFILES:
        lines += ['## ' + TITLES[profile], '']
        found = next((item for item in entries if item['profile'] == profile), None)
        if found is None:
            lines += ['아직 완료된 실행이 없습니다.', ''];continue
        lines += [f'실행 폴더: `{found["run_dir"]}`', '',
                  (root / found['run_dir'] / 'comparison/comparison.md').read_text(), '']
    (root / 'comparison.md').write_text('\n'.join(lines))
    print(f'결과표: {root / "comparison.md"}\nCSV: {root / "comparison.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan');p.add_argument('--profile', choices=(*PROFILES, 'both'), required=True)
    p.add_argument('--image_root', default='data/nuscenes')
    for name in ('start', 'complete', 'fail'):
        p = sub.add_parser(name);p.add_argument('--run_dir', required=True)
        p.add_argument('--profile', choices=PROFILES, required=True)
        if name == 'fail':p.add_argument('--exit_code', type=int, required=True)
    p = sub.add_parser('public_baseline');p.add_argument('--run_dir', required=True)
    p.add_argument('--config', required=True);p.add_argument('--data_path', default='data/drivelm.json')
    p.add_argument('--image_root', default='data/nuscenes')
    p = sub.add_parser('summarize');p.add_argument('--root', default='outputs/report_experiments')
    args = parser.parse_args()
    if args.command == 'plan':
        selected = PROFILES if args.profile == 'both' else [args.profile]
        for profile in selected:
            print(TITLES[profile])
            print('  데이터/환경 준비 → Simple 새 학습 → Reason 새 학습 → 두 모델 추론 → 공통 noGPT 표')
            print('  QA 90/10 · train seed 3407 / eval seed 3047 · sampling' if profile == 'public_baseline'
                  else '  scene 80/10/10 · validation 모델 선택 · greedy')
            print(f'  이미지: {args.image_root}; 출력: outputs/report_experiments/{profile}/<run_id>/comparison/')
        print('CUDA 비교 실험. 이 명령은 설치·다운로드·학습·API 호출을 하지 않았습니다.')
    elif args.command == 'start':start(args.run_dir, args.profile)
    elif args.command == 'fail':state(args.run_dir, profile=args.profile, status='failed', exit_code=args.exit_code)
    elif args.command == 'complete':complete(args.run_dir, args.profile)
    elif args.command == 'summarize':summarize(args.root)
    else:
        try:
            run_public(args)
        except Exception as error:
            state(args.run_dir, status='failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
