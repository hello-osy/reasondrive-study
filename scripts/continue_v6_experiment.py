"""Reuse a finished Simple model and restart an unsaved Reason phase in a new run."""
import argparse
import copy
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from data_utils import write_json
from experiment_config import validate_config
from eval_finetuned_models import validate_training_run
from report_experiment import ROOT, start, state, run_step, complete, summarize


def plan(source):
    source = Path(source).resolve()
    if json.loads((source / 'status.json').read_text()).get('profile') != 'modified':
        raise ValueError('A modified-profile source is required')
    if (source / 'reason/completed.json').exists() or list((source / 'reason').glob('checkpoint-*')):
        raise ValueError('Reason already has saved training progress; refusing to restart it')
    config = json.loads((source / 'simple/training_config.json').read_text())
    if config['mode'] != 'simple':
        raise ValueError('Expected a completed Simple model')
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        if not (source / 'simple' / name).is_file():
            raise ValueError('Missing saved Simple adapter: ' + name)
    manifest = json.loads((source / 'split_manifest.json').read_text())
    original = validate_config(config['experiment'])
    validate_training_run(config, manifest, source / 'simple', original)
    modified = copy.deepcopy(original)
    modified['common'].update(batch_size=30, gradient_accumulation_steps=1)
    modified['description'] = 'Reason 배치 30 재시작. 완료된 Simple은 원래 설정으로 재사용.'
    validate_config(modified)
    data = Path(config['data_path'])
    images = Path(config['base_image_path'])
    data = data if data.is_absolute() else ROOT / data
    images = images if images.is_absolute() else ROOT / images
    if not data.is_file() or not images.is_dir():
        raise ValueError('Recorded data/image paths are unavailable')
    return source, original, modified, data, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source, original, modified, data, images = plan(args.source)
    print(f'Simple 재사용: {source / "simple"}', flush=True)
    print('Reason: 체크포인트 없음. 배치 30·누적 1로 처음부터 학습 → 두 모델 평가 → V6 표 생성', flush=True)
    print('Instruct는 실행하지 않습니다. Simple과 Reason의 배치 차이는 결과에 기록합니다.', flush=True)
    if args.dry_run:
        return
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=source.parent))
    start(run, 'modified')
    try:
        write_json(run / 'experiment.json', modified)
        write_json(run / 'simple_experiment.json', original)
        write_json(run / 'continuation.json', {'source_run': str(source), 'reused_simple': str(source / 'simple'),
                   'reason_restart': 'no checkpoint before interruption', 'reason_batch_size': 30})
        shutil.copy2(source / 'split_manifest.json', run / 'split_manifest.json')
        shutil.copy2(source / 'input_repair_report.json', run / 'input_repair_report.json')
        shutil.copy2(source / 'simple/graph_context.json', run / 'graph_context.json')
        # Keep reused artifacts inside this run for the completion integrity check.
        # The final adapter is sufficient for inference; optimizer checkpoints are not needed.
        shutil.copytree(source / 'simple', run / 'simple', ignore=shutil.ignore_patterns('checkpoint-*'))
        print(f'새 실행 결과: {run}', flush=True)
        run_step(run, 'train_reason', [ROOT / 'scripts/vlm-finetuning.py', '--config', run / 'experiment.json',
                 '--mode', 'reason', '--data_path', data, '--base_image_path', images,
                 '--split_path', run / 'split_manifest.json', '--output_dir', run / 'reason',
                 '--exact_output_dir', '--no-use_wandb'])
        for mode, config in [('simple', 'simple_experiment.json'), ('reason', 'experiment.json')]:
            run_step(run, 'evaluate_' + mode, [ROOT / 'scripts/eval_finetuned_models.py', '--config', run / config,
                     '--model_path', run / mode, '--data_path', data, '--base_image_path', images,
                     '--output_path', run / f'{mode}.json'])
        run_step(run, 'score', [ROOT / 'scripts/evaluate.py', '--responses', f'simple={run}/simple.json',
                 f'reason={run}/reason.json', '--data_path', data, '--output_dir', run / 'comparison',
                 '--allow_batching_difference'])
        complete(run, 'modified')
        summarize(run.parent.parent)
        print(f'V6 표: {run / "comparison/v6_table8.md"}', flush=True)
    except BaseException as error:
        state(run, status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
