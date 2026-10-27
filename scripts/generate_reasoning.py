# ============================================
# [논문 기반 재구현] 파일 전체: ReasonDrive Table 1/§3.1의 reasoning 데이터 생성 절차.
# 공개 생성 코드가 없어 논문을 바탕으로 새로 작성했다. 원저자 구현과 동일하다는 뜻은 아니다.
# provider/model 선택 및 운영 기능은 아래 [실험용 확장·설정 변경] 부분에 별도로 표시한다.
# ============================================
"""[논문 기반 재구현] ReasonDrive Table 1/§3.1: six views + Q + GT → reasoning.

원본 저자의 공개 생성 코드를 복구한 것이 아니라 논문의 절차를 재구현했다.
solution/image/problem/metadata는 보존하고 thinking 및 reasoning_provenance만 쓴다.
입력: 기존 ReasonDrive JSON/JSONL 행 구조. thinking은 없어도 된다.
기본은 dry-run이며 --execute에만 외부 API 요청이 발생한다. GPT 채점은 하지 않는다.

[실험용 확장·설정 변경] provider/model, 출력 한도, 재시도, 재개/출처 기록은
재현 실험을 위한 추가 기능이다. Gemini 결과는 원논문 GPT-4o 결과와 구분한다.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from data_utils import dataset_hash, tagged_text, write_json
from reasoning_providers import ProviderConfig, ProviderError, api_key, generate

PROTOCOL = 'reasondrive-table1-reimplementation-v1'
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
SYSTEM = ("You are assisting in developing a reasoning system for an autonomous driving AI. "
          "Below is a question and answer pair from the '{category}' category. Your task is to "
          "generate a structured, step-by-step reasoning process that logically leads to the provided answer.")
# ============================================
# [논문 기반 재구현] Table 1의 범주별 지침과 §3.1 공통 system prompt.
# 아래 Q/GT 직렬화와 형식 요청은 재구현이며 공개되지 않은 원본 전체 prompt와 같다고 주장하지 않는다.
# ============================================
CATEGORY_PROMPTS = {
    'perception': 'Quickly summarize the observed scene. Identify key objects and their positions. '
                  'Note immediate visual cues and statuses. Format response within <think> tags using 1 concise sentence.',
    'prediction': 'Concisely forecast future states based on current data. Consider object motion, momentum, '
                  'and interactions. Apply basic traffic rules and driver behavior. '
                  'Format response within <think> tags using 1-2 sentences.',
    'planning': 'Assess safety and prioritize actions. Evaluate decision options and trade-offs. '
                'Consider alternative actions and consequences. Format response within <think> tags using 2-3 sentences.',
    'behavior': 'Analyze motion patterns, speed, and trajectories. Consider environmental factors and multi-view '
                'observations. Determine the underlying intent based on dynamic context. '
                'Format response within <think> tags using 1-2 concise sentences.',
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_rows(path):
    text = Path(path).read_text()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Input must contain a nonempty JSON array or JSONL of ReasonDrive rows')
    return rows


def prompts(row):
    category = row['metadata']['category']
    if category not in CATEGORY_PROMPTS:
        raise ValueError('metadata.category must be perception/prediction/planning/behavior')
    if not isinstance(row.get('problem'), str) or not row['problem'].strip():
        raise ValueError('A nonempty problem is required')
    answer = tagged_text(row.get('solution'), 'answer')
    qa = row['metadata'].get('original_qa_pair')
    if qa is not None and (qa.get('Q') != row['problem'] or str(qa.get('A', '')).strip() != answer):
        raise ValueError('Original QA disagrees with problem/solution')
    return (SYSTEM.format(category=category), CATEGORY_PROMPTS[category]
            + '\n\nQuestion: ' + row['problem'] + '\nGround-truth answer: ' + answer
            + '\n\nReturn only the requested <think>...</think> explanation; do not rewrite the answer.')


def image_path(value, root):
    if not isinstance(value, str):
        raise ValueError('image entries must be local file paths')
    if value.startswith('../nuscenes/'):
        return (root / value[len('../nuscenes/'):]).resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def inspect_image(path):
    data = path.read_bytes()
    if data.startswith(b'\xff\xd8\xff'):
        mime = 'image/jpeg'
    elif data.startswith(b'\x89PNG\r\n\x1a\n'):
        mime = 'image/png'
    elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        mime = 'image/webp'
    else:
        raise ValueError(f'Unsupported image signature: {path.name}')
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'mime_type': mime}


def prepare(rows, image_root, config):
    cache, items = {}, []
    root = Path(image_root).resolve()
    for index, row in enumerate(rows):
        item = {'index': index, 'source_row_sha256': digest(row)}
        try:
            system, user = prompts(row)
            paths = row.get('image')
            if not isinstance(paths, list) or len(paths) != 6:
                raise ValueError('Each row must contain exactly six image paths')
            images = []
            for value in paths:
                path = image_path(value, root)
                camera = next((cam for cam in CAMERAS if cam in path.parts or path.name.startswith(cam + '__')), None)
                if camera is None:
                    raise ValueError(f'Cannot identify nuScenes camera: {path.name}')
                if path not in cache:
                    cache[path] = inspect_image(path)
                images.append(dict(cache[path], camera=camera))
            if {image['camera'] for image in images} != set(CAMERAS):
                raise ValueError('Six distinct nuScenes camera views are required')
            # [실험용 확장·설정 변경] 한도 검사는 이미지 전송 없이 미리 수행한다.
            approximate_bytes = sum(4 * ((image['bytes'] + 2) // 3) for image in images) + len(system.encode()) + len(user.encode()) + 4096
            if approximate_bytes > config.max_request_bytes:
                raise ValueError('Encoded request exceeds max_request_bytes; images are not silently resized')
            item.update(system=system, user=user, images=images, status='ready')
            item['request_sha256'] = digest({'system': system, 'user': user, 'images': images, 'config': config.public_dict()})
        except (KeyError, TypeError, AttributeError, ValueError, OSError) as exc:
            item.update(status='invalid_input', error_type=type(exc).__name__, error=str(exc))
        items.append(item)
    return {'protocol': PROTOCOL, 'dataset_sha256': dataset_hash(rows), 'image_root': str(root),
            'config': config.public_dict(), 'prompts_sha256': digest({'system': SYSTEM, 'categories': CATEGORY_PROMPTS}),
            'items': items}


def valid_thinking(text):
    match = re.fullmatch(r'\s*<think>(.*?)</think>\s*', text, re.S)
    if not match or not match.group(1).strip() or re.search(r'</?(?:think|answer)\b', match.group(1), re.I):
        raise ValueError('Expected exactly one nonempty <think> block and no answer block or extra text')
    return '<think>' + match.group(1).strip() + '</think>'


# ============================================
# [실험용 확장·설정 변경] 재개 journal, source/image/config 해시, 실패 상태와 동시 실행 제어.
# 논문에는 이러한 운영 기능의 구체적 구현이 제시되어 있지 않다.
# ============================================
def append_event(path, event):
    # [실험용 확장·설정 변경] 결과를 먼저 fsync해 snapshot 쓰기 중 중단돼도 복구한다.
    import os
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path, signature, count):
    latest = {}
    if not path.exists():
        return latest
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError('Incomplete journal line; preserve the file and inspect before retrying') from None
        index = event.get('index')
        if event.get('run_sha256') != signature or type(index) is not int or not 0 <= index < count:
            raise ValueError('Journal does not match this input/configuration')
        latest[index] = event
    return latest


def snapshot(rows, latest, output, signature):
    generated = []
    counts = {'ok': 0, 'failed': 0, 'pending': 0, 'started': 0}
    for index, source in enumerate(rows):
        row = copy.deepcopy(source)
        event = latest.get(index, {'status': 'pending', 'run_sha256': signature, 'index': index})
        status = event['status']
        counts[status] += 1
        row['thinking'] = event['thinking'] if status == 'ok' else ''
        row['reasoning_provenance'] = {key: value for key, value in event.items() if key != 'thinking'}
        generated.append(row)
    summary = dict(counts, total=len(rows), complete=counts['ok'] == len(rows), run_sha256=signature,
                   output=str(output), updated_at=stamp(),
                   note='Only status=ok rows contain generated thinking. Pending/failed/started rows are not training-ready.')
    write_json(output, generated)
    write_json(str(output) + '.summary.json', summary)
    return summary


def _run(data_path, output_path, image_root, config, *, execute=False, resume=False,
         retry_failed=False, limit=None, generator=generate):
    source, output = Path(data_path).resolve(), Path(output_path).resolve()
    artifacts = [output] + [Path(str(output) + suffix) for suffix in
                            ('.plan.json', '.manifest.json', '.events.jsonl', '.summary.json', '.lock')]
    if any(source == path or (path.exists() and source.samefile(path)) for path in artifacts):
        raise ValueError('Input and output must differ; the original dataset is never overwritten')
    if output.suffix != '.json':
        raise ValueError('Output must use .json (a full row-preserving JSON array)')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    rows = load_rows(source)
    plan = prepare(rows, image_root, config)
    signature = digest(plan)
    if not execute:
        valid = sum(item['status'] == 'ready' for item in plan['items'])
        summary = {'dry_run': True, 'api_calls': 0, 'total': len(rows),
                   'valid': valid,
                   'invalid': sum(item['status'] != 'ready' for item in plan['items']),
                   'max_rows_this_run': limit, 'run_sha256': signature,
                   'max_api_requests_if_executed': min(valid, limit or valid) * config.max_attempts,
                   'plan_path': str(output) + '.plan.json'}
        write_json(summary['plan_path'], dict(plan, summary=summary))
        return summary

    manifest_path, events_path = Path(str(output) + '.manifest.json'), Path(str(output) + '.events.jsonl')
    if resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get('run_sha256') != signature:
            raise ValueError('Resume requires an identical dataset, image contents, prompts and configuration')
    elif any(path.exists() for path in (output, manifest_path, events_path, Path(str(output) + '.summary.json'))):
        raise FileExistsError('Output artifacts already exist; use --resume or a new output filename')
    latest = read_events(events_path, signature, len(rows))
    needs_api = any(item['status'] == 'ready' and (item['index'] not in latest or
                    (retry_failed and latest[item['index']]['status'] != 'ok')) for item in plan['items'])
    if needs_api:
        api_key(config)
    if not resume:
        write_json(manifest_path, dict(plan, run_sha256=signature, created_at=stamp(), source_path=str(source)))
    submitted = 0
    for item in plan['items']:
        index = item['index']
        previous = latest.get(index)
        if previous and (previous['status'] == 'ok' or not retry_failed):
            continue
        event = {'index': index, 'run_sha256': signature, 'source_row_sha256': item['source_row_sha256'],
                 'requested_model': config.model, 'provider': config.provider, 'started_at': stamp(),
                 'row_attempt': (previous or {}).get('row_attempt', 0) + 1,
                 'dataset_sha256': plan['dataset_sha256'], 'prompts_sha256': plan['prompts_sha256']}
        if item['status'] != 'ready':
            event.update(status='failed', error_type=item['error_type'], error=item['error'], attempts=0)
            append_event(events_path, event)
            latest[index] = event
            continue
        if limit is not None and submitted >= limit:
            continue
        event.update(status='started', request_sha256=item['request_sha256'], images=item['images'])
        append_event(events_path, event)
        latest[index] = event
        submitted += 1
        stop_for_configuration_error = False
        try:
            images = []
            for info in item['images']:
                data = Path(info['path']).read_bytes()
                if hashlib.sha256(data).hexdigest() != info['sha256']:
                    raise ValueError('Image contents changed after preflight; no API request was made')
                images.append(dict(info, data=data))
            result = generator(config, item['system'], item['user'], images)
            event.update({key: value for key, value in result.items() if key != 'text'})
            event['raw_text'] = result['text']
            if not result['complete']:
                raise ValueError('API output was incomplete, refused or blocked')
            event.update(thinking=valid_thinking(result['text']), status='ok')
        except (ProviderError, ValueError, OSError) as exc:
            event.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            if isinstance(exc, ProviderError):
                event['attempts'] = exc.attempts
                event['http_status'] = exc.status
                stop_for_configuration_error = exc.status in (400, 401, 403, 404, 422)
        event['completed_at'] = stamp()
        append_event(events_path, event)
        latest[index] = event
        snapshot(rows, latest, output, signature)
        if stop_for_configuration_error:
            break
    return snapshot(rows, latest, output, signature)


def run(data_path, output_path, image_root, config, *, execute=False, resume=False,
        retry_failed=False, limit=None, generator=generate):
    if not execute:
        return _run(data_path, output_path, image_root, config, execute=False,
                    resume=resume, retry_failed=retry_failed, limit=limit, generator=generator)
    # [실험용 확장·설정 변경] 같은 output에 대한 동시 과금·journal 혼합을 막는다 (macOS/Linux).
    import fcntl
    lock_path = Path(str(Path(output_path).resolve()) + '.lock')
    source = Path(data_path).resolve()
    if source == lock_path or (lock_path.exists() and source.samefile(lock_path)):
        raise ValueError('Input and output must differ; the original dataset is never overwritten')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process is using this output; choose another output or wait') from None
        try:
            return _run(data_path, output_path, image_root, config, execute=True,
                        resume=resume, retry_failed=retry_failed, limit=limit, generator=generator)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--image_root', required=True, help='nuScenes root replacing ../nuscenes in input image paths')
    parser.add_argument('--config', help='Provider JSON configuration; no API keys in this file')
    parser.add_argument('--provider', choices=['openai', 'gemini'])
    parser.add_argument('--model', help='Explicit API model ID; never silently substituted')
    parser.add_argument('--api_key_env', help='Environment variable name, never the key itself')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Send billable API requests; default is dry-run')
    mode.add_argument('--dry_run', action='store_true', help='Validate and write only *.plan.json; no API/key needed')
    parser.add_argument('--resume', action='store_true', help='Reuse successful journal rows after checking all hashes')
    parser.add_argument('--retry_failed', action='store_true', help='With --resume, retry failures/interrupted calls (may charge again)')
    parser.add_argument('--limit', type=int, help='Maximum newly submitted rows, for a small pilot or incremental resume')
    args = parser.parse_args()
    try:
        options = json.loads(Path(args.config).read_text()) if args.config else {}
        if not isinstance(options, dict):
            raise ValueError('Configuration must be a JSON object')
        options.pop('_comment', None)
        for name in ('provider', 'model', 'api_key_env'):
            if getattr(args, name) is not None:
                options[name] = getattr(args, name)
        config = ProviderConfig(**options)
        if args.retry_failed and not args.resume:
            raise ValueError('--retry_failed requires --resume')
        summary = run(args.data_path, args.output_path, args.image_root, config,
                      execute=args.execute, resume=args.resume, retry_failed=args.retry_failed, limit=args.limit)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get('invalid', 0) or (not summary.get('dry_run') and not summary['complete']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
