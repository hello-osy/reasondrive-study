# ============================================
# 실험용 확장·설정 변경 — 파일 전체
# 분할·입력·출력 처리를 공통화한 보조 모듈. 개별 변경의 근거는 아래에 표시한다.
# ============================================
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


# ============================================
# 공개 코드 오류 수정 — 해당 부분
# 학습·평가에서 서로 다른 seed로 재분할하던 문제를 저장 manifest 재사용으로 해결.
# 장면 단위 80/10/10 분할 자체는 C1을 위한 실험 설정이다.
# ============================================
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


# ============================================
# 논문 기반 재구현: 생성 답변 parser의 구조 검증 보완(재구현부 수정)
# ============================================
def generated_answer(text):
    """재구현부 보완: 생성 답변의 균형·중복·중첩을 검사한다; 원문 복구와 분리."""
    text = str(text or '').strip()
    tokens = list(re.finditer(r'</?(?:think|answer)>', text))
    stripped = re.sub(r'</?(?:think|answer)>', '', text)
    if re.search(r'</?\s*(?:think|answer)\b', stripped):
        raise ValueError('Malformed generated think/answer tag')
    active, start, answer = None, None, None
    cursor = 0
    for token in tokens:
        if active is None and text[cursor:token.start()].strip():
            raise ValueError('Generated prose must be inside think/answer tags')
        tag = token.group()
        closing = tag.startswith('</')
        name = tag[2:-1] if closing else tag[1:-1]
        if not closing:
            if active is not None or answer is not None:
                raise ValueError('Nested, duplicate or misplaced generated answer/think tag')
            active, start = name, token.end()
        else:
            if active != name:
                raise ValueError('Unbalanced generated answer/think tags')
            if name == 'answer':
                answer = text[start:token.start()].strip()
                if not answer:
                    raise ValueError('Empty generated answer')
            active = None
        cursor = token.end()
    if active is not None or answer is None or text[cursor:].strip():
        raise ValueError('Expected one complete final <answer> pair')
    return answer


def tagged_text(text, tag, required=False):
    # Existing inference callers keep their API; permissive source parsing stays separate.
    if tag == 'answer' and required:
        return generated_answer(text)
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


# ============================================
# 실험용 확장·설정 변경 — 해당 부분
# 로컬 전처리에서 제외되던 원본 설명을 정규화한다. 생성 답변 복구에는 사용하지 않는다.
# ============================================
def normalize_reasoning(text):
    """Normalize source annotations only; preserve prose, coordinates and ordering.

    This is deliberately separate from strict generated-answer parsing. The source
    contains mixed tagged/plain prose, unclosed tags and a '<thinkThe' typo.
    """
    text = str(text or '').replace('\\n', '\n').strip()
    text = re.sub(r'<think(?=[A-Z])', '', text)
    text = re.sub(r'</?think>', '\n\n', text)
    text = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', text).strip()
    if not text:
        raise ValueError('Empty think')
    if re.search(r'</?think', text):
        raise ValueError('Unrecognized thinking tag')
    return text


def clean_message_content(messages):
    """Remove Arrow's null union fields before Qwen detects image entries."""
    from urllib.parse import urlsplit, unquote
    cleaned = []
    for message in messages:
        message = dict(message)
        if isinstance(message.get('content'), list):
            parts = []
            for part in message['content']:
                part = {k: v for k, v in part.items() if v is not None}
                image = part.get('image')
                if isinstance(image, str) and image.startswith('file://'):
                    uri = urlsplit(image)
                    if uri.netloc not in ('', 'localhost'):
                        raise ValueError('Expected a local image URI')
                    part['image'] = unquote(uri.path)
                parts.append(part)
            message['content'] = parts
        cleaned.append(message)
    return cleaned



# ============================================
# 논문 기반 재구현 — 해당 부분
# 원본 Reason 입력 형식을 유지하면서 논문의 Simple 비교 모드를 추가했다.
# ============================================
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
        thinking = normalize_reasoning(row.get('thinking'))
        answer = tagged_text(row.get('solution'), 'answer')
        completion = f'<answer>{answer}</answer>'
        if mode == 'reason':
            completion = f'<think>{thinking}</think>\n\n{completion}'
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': completion}]})
    return {'messages': messages}
