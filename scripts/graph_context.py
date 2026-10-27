# ============================================
# 실험용 확장·설정 변경 — 파일 전체
# 원본 root를 복원하여 채점 없는 별도 추론 문맥으로 사용한다.
# ============================================
"""평가 그래프의 누락 시작 질문을 원본 DriveLM annotation에서 복원한다.

논문 기반 재구현: 원본 객체 질문과 좌표 그래프를 연결한다.
재구현부 보완: 누락 root를 별도로 추론하며 원래 test 문항 수는 유지한다.
실험용 확장·설정 변경: 복원한 질문은 graph context로만 사용하고 채점하지 않는다.
"""
import hashlib
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

from data_utils import dataset_hash, write_json

ROOT_QUESTION = ('What are the important objects in the current scene? Those objects will be '
                 'considered for the future reasoning and driving decision.')
ANNOTATION_REVISION = 'ae973fbd4e8d4684af4ab234d504bd6c5e946868'
ANNOTATION_NAME = 'v1_1_train_nus.json'
ANNOTATION_URL = (f'https://huggingface.co/datasets/OpenDriveLab/DriveLM/resolve/'
                  f'{ANNOTATION_REVISION}/{ANNOTATION_NAME}')
GOOGLE_ID = '1CvTPwChKvfnvrZ1Wr0ZNVqtibkkNeGgt'
# The official HF and Google Drive mirrors are linked in the pinned DriveLM README.
ANNOTATION_SHA256 = 'ecd3c13e9beb27295bcf6182ea35fff7c1d6a9ba58d40e48fb38d713094759fd'
MAX_ANNOTATION_BYTES = 210 * 1024 * 1024


def graph_context_path(split_manifest_path):
    return Path(split_manifest_path).parent / 'graph_context.json'


def _frames(rows, indices):
    frames = {}
    for index in indices:
        row = rows[index]
        meta = row.get('metadata') or {}
        if not meta.get('scene_id') or not meta.get('frame_id'):
            raise ValueError(f'Row {index}: scene_id/frame_id required for graph preflight')
        qa = meta.get('original_qa_pair') or {}
        if qa.get('Q') != row.get('problem') or not isinstance(qa.get('A'), str):
            raise ValueError(f'Row {index}: original QA required for graph preflight')
        if not isinstance(qa.get('tag'), list) or not qa['tag']:
            raise ValueError(f'Row {index}: original QA tags required for graph preflight')
        key = (meta['scene_id'], meta['frame_id'])
        frames.setdefault(key, []).append(index)
    return frames


def root_index(rows, indices):
    roots = [i for i in indices if rows[i]['problem'] == ROOT_QUESTION]
    if len(roots) > 1:
        raise ValueError('Duplicate object-list root in one frame')
    if roots:
        meta = rows[roots[0]]['metadata']
        if meta.get('category') != 'perception' or 2 not in meta['original_qa_pair']['tag']:
            raise ValueError('Object-list root must be perception with language tag 2')
        return roots[0]
    return None


def missing_root_frames(rows, indices):
    return {key: ids for key, ids in _frames(rows, indices).items()
            if root_index(rows, ids) is None}


def _image_list(row):
    images = row['image']
    return list(images) if isinstance(images, list) else [images]


def _image_identity(path):
    # Original annotations and ReasonDrive use different nuScenes directory prefixes.
    return str(path).split('/samples/', 1)[-1]


def _digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class _DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.params = None, {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form':
            self.action = attrs.get('action')
        if tag == 'input' and attrs.get('type') == 'hidden':
            self.params[attrs['name']] = attrs.get('value', '')


def _download(url, target, confirmations=0):
    with urlopen(url, timeout=60) as response:
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            if confirmations >= 1:
                raise ValueError('Official annotation mirror did not provide the download after confirmation')
            form = _DownloadForm()
            form.feed(response.read(64 * 1024).decode())
            host = urlsplit(form.action or '').hostname
            if (urlsplit(form.action or '').scheme != 'https'
                    or host != 'drive.usercontent.google.com' or form.params.get('id') != GOOGLE_ID):
                raise ValueError('Official annotation mirror returned an unexpected HTML page')
            return _download(form.action + '?' + urlencode(form.params), target, confirmations + 1)
        with Path(target).open('wb') as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ANNOTATION_BYTES:
                    raise ValueError('DriveLM annotation exceeds the expected 210 MiB limit')
                output.write(chunk)


def official_annotations(cache_path):
    """Download only the 193 MB (184 MiB) annotation, never images or model weights."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    path = Path(cache_path)
    if path.is_file() and _digest_file(path) == ANNOTATION_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    failures = []
    print('그래프 문맥 준비: 공식 DriveLM annotation 약 193 MB 다운로드/검증')
    urls = [ANNOTATION_URL, f'https://drive.google.com/uc?export=download&id={GOOGLE_ID}']
    for url in urls:
        try:
            if url == ANNOTATION_URL:
                # Hub client uses the configured login token and shared download cache.
                cached = hf_hub_download('OpenDriveLab/DriveLM', ANNOTATION_NAME,
                                         repo_type='dataset', revision=ANNOTATION_REVISION)
                if Path(cached).stat().st_size > MAX_ANNOTATION_BYTES:
                    raise ValueError('DriveLM annotation exceeds the expected 210 MiB limit')
                shutil.copyfile(cached, temporary)
            else:
                _download(url, temporary)
            if _digest_file(temporary) != ANNOTATION_SHA256:
                raise ValueError('Official annotation SHA-256 mismatch')
            temporary.replace(path)
            return path
        except (OSError, ValueError, HfHubHTTPError) as error:
            failures.append(str(error))
            temporary.unlink(missing_ok=True)
    raise ValueError('Cannot restore graph roots before training. Provide --drivelm_annotations '
                     'with the original v1_1_train_nus.json. ' + '; '.join(failures))


def build_graph_context(rows, manifest, annotations=None, source=None):
    """Build context-only input records; source answers never enter inference prompts."""
    if manifest.get('dataset_sha256') != dataset_hash(rows):
        raise ValueError('Graph preflight dataset differs from the split manifest')
    items = []
    missing = missing_root_frames(rows, manifest['indices']['test'])
    if missing and annotations is None:
        raise ValueError(f'{len(missing)} test frames need original DriveLM root annotations')
    for (scene, frame), ids in missing.items():
        try:
            original = annotations[scene]['key_frames'][frame]
            candidates = [qa for qa in original['QA']['perception']
                          if qa['Q'] == ROOT_QUESTION]
        except (KeyError, TypeError) as error:
            raise ValueError(f'Original annotation missing frame {scene}/{frame}') from error
        if len(candidates) != 1 or not str(candidates[0].get('A', '')).strip():
            raise ValueError(f'Original annotation needs exactly one root QA: {scene}/{frame}')
        images = _image_list(rows[ids[0]])
        if any(_image_list(rows[i]) != images for i in ids):
            raise ValueError(f'Inconsistent images within frame {scene}/{frame}')
        original_images = original.get('image_paths')
        if not isinstance(original_images, dict) or set(map(_image_identity, original_images.values())) != set(map(_image_identity, images)):
            raise ValueError(f'Original/source image identity differs: {scene}/{frame}')
        qa = {'Q': ROOT_QUESTION, 'A': candidates[0]['A'], 'tag': [2]}
        items.append({'id': f'graph:{scene}:{frame}', 'problem': ROOT_QUESTION,
                      'solution': f"<answer>{qa['A']}</answer>", 'image': images,
                      'metadata': {'scene_id': scene, 'frame_id': frame, 'category': 'perception',
                                   'original_qa_pair': qa}, 'scored': False})
    payload = {'version': 1, 'dataset_sha256': dataset_hash(rows),
               'test_indices': manifest['indices']['test'], 'source': source, 'items': items}
    payload['sha256'] = dataset_hash(payload)
    return validate_graph_context(payload, rows, manifest['indices']['test'])


def validate_graph_context(payload, rows, test_indices):
    if payload.get('version') != 1 or payload.get('dataset_sha256') != dataset_hash(rows):
        raise ValueError('Graph context source dataset/version mismatch')
    if payload.get('test_indices') != list(test_indices):
        raise ValueError('Graph context test indices mismatch')
    if payload.get('sha256') != dataset_hash({k: v for k, v in payload.items() if k != 'sha256'}):
        raise ValueError('Graph context SHA-256 mismatch')
    missing = missing_root_frames(rows, test_indices)
    seen = set()
    for item in payload.get('items', []):
        meta = item.get('metadata') or {}
        key = (meta.get('scene_id'), meta.get('frame_id'))
        if key not in missing or key in seen or item.get('id') != f'graph:{key[0]}:{key[1]}':
            raise ValueError('Duplicate, unexpected or misidentified graph context')
        qa = meta.get('original_qa_pair') or {}
        if (item.get('scored') is not False or item.get('problem') != ROOT_QUESTION
                or meta.get('category') != 'perception'
                or qa.get('Q') != ROOT_QUESTION or qa.get('tag') != [2]
                or not isinstance(qa.get('A'), str) or not qa['A'].strip()
                or item.get('solution') != f"<answer>{qa['A']}</answer>"
                or _image_list(item) != _image_list(rows[missing[key][0]])):
            raise ValueError('Invalid context-only root record')
        seen.add(key)
    if seen != set(missing):
        raise ValueError('Missing graph contexts; do not drop incomplete frames')
    if missing and not (payload.get('source') or {}).get('sha256'):
        raise ValueError('Original annotation provenance required for restored roots')
    return payload


def prepare_graph_context(rows, manifest, split_manifest_path, annotations_path=None):
    path = graph_context_path(split_manifest_path)
    if path.is_file() and annotations_path is None:
        return load_graph_context(split_manifest_path, rows, manifest)
    missing = missing_root_frames(rows, manifest['indices']['test'])
    annotations, source = None, None
    if missing:
        path_source = (Path(annotations_path) if annotations_path else
                       official_annotations(path.parent / ANNOTATION_NAME))
        digest = _digest_file(path_source)
        # User-supplied annotations are permitted but must match the pinned official artifact.
        if digest != ANNOTATION_SHA256:
            raise ValueError('Original DriveLM annotation does not match the pinned SHA-256')
        annotations = json.loads(path_source.read_text())
        source = {'repo': 'OpenDriveLab/DriveLM', 'revision': ANNOTATION_REVISION,
                  'filename': ANNOTATION_NAME, 'sha256': digest,
                  'url': f'https://drive.google.com/uc?export=download&id={GOOGLE_ID}',
                  'huggingface_url': ANNOTATION_URL, 'google_drive_id': GOOGLE_ID,
                  'verification': 'SHA-256 pinned from the official Google Drive annotation mirror'}
    payload = build_graph_context(rows, manifest, annotations, source)
    write_json(path, payload)
    return payload


def load_graph_context(split_manifest_path, rows, manifest):
    path = graph_context_path(split_manifest_path)
    if not path.is_file():
        raise ValueError(f'Missing graph context preflight: {path}. Run prepare_inputs.py before training/inference.')
    return validate_graph_context(json.loads(path.read_text()), rows, manifest['indices']['test'])
