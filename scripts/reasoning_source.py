"""Strict train-only reasoning overlays; original rows and evaluation hashes stay intact."""
import copy
import hashlib
import json
from pathlib import Path
from data_utils import dataset_hash, normalize_reasoning


def load_reasoning_source(path, rows, manifest):
    if not path or str(path) == 'original':
        return {}, {'kind': 'original'}
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('schema_version') != 1 or data.get('kind') != 'train_reasoning_overlay':
        raise ValueError('Expected a train_reasoning_overlay file')
    if data.get('dataset_sha256') != dataset_hash(rows):
        raise ValueError('Reasoning source dataset hash mismatch')
    if data.get('split_manifest') != manifest:
        raise ValueError('Reasoning source split mismatch; use its original scene split/settings')
    train = set(manifest['indices']['train'])
    excluded = set()
    if data.get('selection_policy') is not None:
        if data['selection_policy'] != 'exclude_conflicts':
            raise ValueError('Unknown reasoning selection policy')
        for record in data.get('excluded_records', []):
            index = record['id']
            if type(index) is not int or index not in train or index in excluded:
                raise ValueError('Invalid excluded training ID')
            if record.get('evidence_status') != 'conflict' or record.get('source_row_sha256') != dataset_hash([rows[index]]):
                raise ValueError('Excluded row must be a matching conflict record')
            excluded.add(index)
    elif data.get('excluded_records'):
        raise ValueError('Exclusions require an explicit selection policy')
    mapping = {}
    for record in data['records']:
        index = record['id']
        if type(index) is not int or index not in train or index in mapping or index in excluded:
            raise ValueError('Overlay contains duplicate, invalid or non-train IDs')
        if record.get('status') != 'ok':
            raise ValueError('Overlay has failed/unreviewed rows')
        if record.get('source_row_sha256') != dataset_hash([rows[index]]):
            raise ValueError(f'Overlay row identity mismatch: {index}')
        mapping[index] = '<think>' + normalize_reasoning(record['thinking']) + '</think>'
    if not mapping or set(mapping) != train - excluded or not data.get('complete'):
        raise ValueError('Reasoning source is incomplete; no silent fallback to original reasoning')
    return mapping, {'kind': 'train_reasoning_overlay', 'path': str(path),
                     'sha256': hashlib.sha256(raw).hexdigest(), 'train_count': len(mapping),
                     'excluded_train_indices': sorted(excluded),
                     'generation_manifest': data.get('generation_manifest'),
                     'semantic_review': data.get('semantic_review', 'not_independently_verified')}


def training_row(row, index, split, mapping):
    if split != 'train' or not mapping:
        return row
    result = copy.deepcopy(row)
    result['thinking'] = mapping[index]
    return result
