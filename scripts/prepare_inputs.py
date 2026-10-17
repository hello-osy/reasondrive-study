"""Audit the automatic input repairs used by both CUDA and MLX before training."""
import argparse
import json
from pathlib import Path
from data_utils import (clean_message_content, format_sample, load_split,
                        normalize_reasoning, tagged_text, write_json)


def prepare_inputs(data_path, image_root, split_path, report_path):
    rows = json.loads(Path(data_path).read_text())
    manifest = load_split(rows, split_path, create=True)
    report = {'dataset_sha256': manifest['dataset_sha256'], 'format_version': 2,
              'counts': {}, 'repaired': [], 'errors': []}
    for split, indices in manifest['indices'].items():
        valid = 0
        for index in indices:
            row = rows[index]
            try:
                # The same functions are used again at training/inference time;
                # source annotations, row IDs and the split hash remain untouched.
                for mode in ('simple', 'reason'):
                    sample = format_sample(row, image_root, mode, training=split != 'test')
                    clean_message_content(sample['messages'])
                tagged_text(row.get('solution'), 'answer')
                if split != 'test':
                    normalized = normalize_reasoning(row.get('thinking'))
                    try:
                        previous = tagged_text(row.get('thinking'), 'think')
                    except ValueError:
                        previous = None
                    if previous != normalized:
                        report['repaired'].append({'index': index, 'split': split,
                                                   'repair': 'normalize_source_reasoning'})
                valid += 1
            except (ValueError, KeyError, TypeError, OSError) as error:
                report['errors'].append({'index': index, 'split': split, 'error': str(error)})
        report['counts'][split] = valid
    write_json(report_path, report)
    print(f"입력 자동 복구·검사: {report['counts']}; 설명 정규화 {len(report['repaired'])}행; "
          f"복구 불가 {len(report['errors'])}행. 보고서: {report_path}")
    if report['errors']:
        raise ValueError(f'Unrepairable inputs; inspect {report_path}. Training has not started.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', default='data/drivelm.json')
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--split_path', default='splits/drivelm.json')
    parser.add_argument('--report_path', required=True)
    args = parser.parse_args()
    try:
        prepare_inputs(args.data_path, args.image_root, args.split_path, args.report_path)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
