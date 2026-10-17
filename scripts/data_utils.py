"""Shared, dependency-free data contract for paired Simple/Reason experiments."""
import hashlib
import json
import random
import re
from pathlib import Path


def dataset_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def scene_id(row):
    value = (row.get('metadata') or {}).get('scene_id')
    if value is None or str(value).strip() in ('', 'unknown'):
        raise ValueError('Every row needs metadata.scene_id for scene-disjoint splitting')
    return str(value)


def load_split(rows, path, seed=3407, val_size=0.1, test_size=0.1, create=False):
    """Persist exact row indices; reject changed data, settings, or leaking manifests."""
    path = Path(path)
    fingerprint = dataset_hash(rows)
    groups = sorted({scene_id(row) for row in rows})
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError('Require positive validation/test fractions summing to less than 1')
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        if not create:
            raise FileNotFoundError(f'Training split manifest required: {path}')
        random.Random(seed).shuffle(groups)
        ntest, nval = max(1, round(len(groups) * test_size)), max(1, round(len(groups) * val_size))
        if ntest + nval >= len(groups):
            raise ValueError('Not enough scenes for nonempty train/validation/test splits')
        scene_sets = {'test': set(groups[:ntest]), 'validation': set(groups[ntest:ntest+nval]),
                      'train': set(groups[ntest+nval:])}
        manifest = {'dataset_sha256': fingerprint, 'seed': seed, 'val_size': val_size,
                    'test_size': test_size, 'indices': {
                        name: [i for i, row in enumerate(rows) if scene_id(row) in ids]
                        for name, ids in scene_sets.items()}}
    for key, expected in [('dataset_sha256', fingerprint), ('seed', seed), ('val_size', val_size), ('test_size', test_size)]:
        if manifest.get(key) != expected:
            raise ValueError(f'Split manifest mismatch: {key}')
    indices = manifest['indices']
    flat = [i for name in ('train', 'validation', 'test') for i in indices[name]]
    if any(type(i) is not int for i in flat) or sorted(flat) != list(range(len(rows))):
        raise ValueError('Split indices must cover each data row exactly once')
    seen = set()
    for name in ('train', 'validation', 'test'):
        scenes = {scene_id(rows[i]) for i in indices[name]}
        if not scenes or seen & scenes:
            raise ValueError('Empty split or scene leakage in split manifest')
        seen.update(scenes)
    if not path.exists():
        write_json(path, manifest)
    return manifest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def tagged_text(text, tag, required=False):
    text = str(text or '').strip()
    matches = re.findall(fr'<{tag}>(.*?)</{tag}>', text, re.S)
    if tag == 'think' and matches and all(part.strip() for part in matches):
        remainder = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
        if not remainder:
            return '\n\n'.join(part.strip() for part in matches)
    if len(matches) == 1 and matches[0].strip():
        return matches[0].strip()
    if required or f'<{tag}>' in text or f'</{tag}>' in text:
        raise ValueError(f'Expected one nonempty <{tag}> pair')
    if not text:
        raise ValueError(f'Empty {tag}')
    return text


def format_sample(row, base_image_path, mode, training=True):
    if mode not in ('simple', 'reason'):
        raise ValueError('mode must be simple or reason')
    paths = row['image'] if isinstance(row['image'], list) else [row['image']]
    if not paths:
        raise ValueError('No images')
    content = []
    for item in paths:
        path = Path(item.replace('../nuscenes', str(base_image_path))).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        content.append({'type': 'image', 'image': path.as_uri(), 'resized_height': 224, 'resized_width': 224})
    question = str(row['problem']).strip()
    if not question:
        raise ValueError('Empty question')
    instruction = ('Provide reasoning within <think></think> and your final answer within <answer></answer>.'
                   if mode == 'reason' else 'Provide only your final answer within <answer></answer>.')
    content.append({'type': 'text', 'text': f'You are an AI assistant for autonomous driving.\n\n{question}\n\n{instruction}'})
    messages = [{'role': 'user', 'content': content}]
    if training:
        # Validate reasoning for BOTH modes so their eligible training rows stay identical.
        thinking = tagged_text(row.get('thinking'), 'think')
        answer = tagged_text(row.get('solution'), 'answer')
        completion = f'<answer>{answer}</answer>'
        if mode == 'reason':
            completion = f'<think>{thinking}</think>\n\n{completion}'
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': completion}]})
    return {'messages': messages}
