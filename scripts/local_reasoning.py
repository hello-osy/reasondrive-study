#!/usr/bin/env python3
"""One-command local six-camera rationale generation. Never edits the source dataset."""
import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import socket
import subprocess
import tarfile
import time
from pathlib import Path
from urllib import request, error

from data_utils import dataset_hash, load_split, tagged_text
from generate_reasoning import load_rows, image_path, CAMERAS, append_event

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = '''You write short, visually grounded explanations for driving QA training.
The six images are different cameras at essentially one instant, NOT a time sequence.
Separate visible observations from inference. Never claim measured speed, acceleration,
or movement between frames. Camera pixel coordinates are not world coordinates.
Respect whether the question asks for a safe action or an action causing a collision.
The provided answer is a dataset label, not proof of a visible fact. Do not invent
objects, traffic lights, lane positions or temporal evidence to justify it.
If speed or another fact is not observable, explicitly state that limitation.
If the image clearly contradicts the label, mark conflict instead of rationalizing it.
Return a final JSON object only: {"explanation": "1-3 concise English sentences, at most 180 words",
"evidence_status": "grounded OR limited OR conflict"}. Do not change the answer.
Do not include think/answer tags in the explanation. Do not output coordinates unless needed.'''


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def web_json(url, payload=None, timeout=60):
    body = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(url, data=body, headers={'Content-Type': 'application/json', 'User-Agent': 'ReasonDrive-local-generator'})
    with request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def validate_config(c):
    for key in ('context_size', 'fit_margin_mib', 'threads', 'threads_batch', 'batch_size',
                'ubatch_size', 'image_max_tokens', 'max_tokens', 'timeout_seconds',
                'startup_timeout_seconds', 'attempts', 'benchmark_samples'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['cache_type'] not in ('f16', 'q8_0') or c['flash_attention'] not in ('on', 'off'):
        raise ValueError('Unsupported cache/attention setting')
    if c['max_tokens'] >= c['context_size'] or not (c['reasoning_budget'] == -1 or 0 < c['reasoning_budget'] < c['max_tokens']):
        raise ValueError('Keep room for images/input and final JSON after the reasoning budget')
    return c


def download(url, dest, expected=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not expected or file_sha(dest) == expected):
        return
    temp = dest.with_suffix(dest.suffix + '.download')
    with request.urlopen(url, timeout=120) as source, temp.open('wb') as target:
        shutil.copyfileobj(source, target)
    if expected and file_sha(temp) != expected:
        raise ValueError('Download checksum mismatch')
    temp.replace(dest)


def safe_extract(archive, destination):
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.isdev() or member.isfifo():
                raise ValueError('Unsafe archive member')
            if member.issym() or member.islnk():
                linked = (target.parent / member.linkname).resolve() if member.issym() else (destination / member.linkname).resolve()
                if not linked.is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe archive symlink')
        bundle.extractall(destination, filter='data')


def bootstrap(c, out, server_override=None):
    """Pin remote refs on first run; use the same resolved artifacts after interruption."""
    path = out / 'runtime.json'
    if path.exists():
        state = json.loads(path.read_text())
        if state['request'] != {k: c[k] for k in ('model_repo', 'model_revision', 'model_file', 'projector_file', 'llama_release')}:
            raise ValueError('Runtime request changed; choose a new output directory')
        if state.get('server_override') != (str(Path(server_override).resolve()) if server_override else None):
            raise ValueError('Server override changed')
    else:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(c['model_repo'], revision=c['model_revision']).sha
        release = None
        if not server_override:
            suffix = 'latest' if c['llama_release'] == 'latest' else 'tags/' + c['llama_release']
            release = web_json('https://api.github.com/repos/ggml-org/llama.cpp/releases/' + suffix)
        if release:
            ref = web_json('https://api.github.com/repos/ggml-org/llama.cpp/commits/' + release['tag_name'])
            release['resolved_commit'] = ref['sha']
        state = {'request': {k: c[k] for k in ('model_repo', 'model_revision', 'model_file', 'projector_file', 'llama_release')},
                 'model_commit': revision, 'release': release,
                 'server_override': str(Path(server_override).resolve()) if server_override else None}
        atomic(path, state)
    from huggingface_hub import hf_hub_download
    models = []
    for name in (c['model_file'], c['projector_file']):
        models.append(Path(hf_hub_download(c['model_repo'], name, revision=state['model_commit'],
                                         cache_dir=ROOT / '.cache/local_reasoning/models')))
    env = dict(os.environ)
    if server_override:
        server = Path(server_override).resolve()
        if not server.is_file():
            raise FileNotFoundError(server)
        libraries = []
    else:
        release = state['release']
        candidates = [a for a in release['assets'] if re.search(r'(ubuntu|linux).*x64.*cuda', a['name'], re.I)
                      and a['name'].endswith(('.tar.gz', '.tar.xz')) and 'cudart' not in a['name']]
        destination = ROOT / '.tools/local_reasoning' / release['tag_name']
        if not candidates:
            build_cuda(release, destination, out)
        asset = sorted(candidates, key=lambda a: a['name'])[-1] if candidates else None
        assets = ([asset] + [a for a in release['assets'] if 'cudart' in a['name'] and
                           re.search(r'(ubuntu|linux).*x64.*cuda', a['name'], re.I) and
                           a['name'].endswith(('.tar.gz', '.tar.xz')) and
                           a['name'].split('cuda-')[-1] == asset['name'].split('cuda-')[-1]]) if asset else []
        for a in assets:
            archive = ROOT / '.cache/local_reasoning/releases' / a['name']
            expected = (a.get('digest') or '').removeprefix('sha256:') or None
            download(a['browser_download_url'], archive, expected)
            marker = destination / (a['name'] + '.extracted')
            if not marker.exists():
                destination.mkdir(parents=True, exist_ok=True)
                safe_extract(archive, destination)
                marker.write_text(file_sha(archive))
        servers = sorted(destination.rglob('llama-server'))
        if len(servers) != 1:
            raise RuntimeError('CUDA archive must contain exactly one llama-server')
        server = servers[0]
        libraries = sorted({str(p.parent) for p in destination.rglob('*.so*')})
        env['LD_LIBRARY_PATH'] = ':'.join(libraries + [env.get('LD_LIBRARY_PATH', '')])
    version = subprocess.check_output([str(server), '--version'], env=env, stderr=subprocess.STDOUT, text=True)
    hashes = {str(p): file_sha(p) for p in [*models, server]}
    if state.get('artifact_hashes') and state['artifact_hashes'] != hashes:
        raise ValueError('Runtime/model files changed since first run')
    state.update(artifact_hashes=hashes, server_version=version, library_paths=libraries)
    atomic(path, state)
    return server, models, env, state


def build_cuda(release, destination, out):
    """Build a pinned official source release with the installed CUDA toolkit."""
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _build_cuda(release, destination, out)


def _build_cuda(release, destination, out):
    server = destination / 'build/bin/llama-server'
    if server.exists():
        return
    nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    if not Path(nvcc).is_file() or not shutil.which('cmake'):
        raise RuntimeError('CUDA toolkit (nvcc) and cmake are required for automatic CUDA build; alternatively pass --server-binary.')
    commit = release['resolved_commit']
    archive = ROOT / '.cache/local_reasoning/releases' / (commit + '.tar.gz')
    print('공식 llama.cpp CUDA 소스 빌드 중 (최초 1회)', flush=True)
    download('https://codeload.github.com/ggml-org/llama.cpp/tar.gz/' + commit, archive)
    destination.mkdir(parents=True, exist_ok=True)
    safe_extract(archive, destination)
    source = destination / ('llama.cpp-' + commit)
    commands = [
        ['cmake', '-S', str(source), '-B', str(destination/'build'), '-DGGML_CUDA=ON',
         '-DCMAKE_CUDA_COMPILER='+nvcc, '-DCMAKE_CUDA_ARCHITECTURES=native', '-DLLAMA_CURL=OFF',
         '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=ON'],
        ['cmake', '--build', str(destination/'build'), '--config', 'Release', '-j', '6', '--target', 'llama-server'],
    ]
    with (out/'build.log').open('a') as log:
        for command in commands:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def server_args(server, models, c, port):
    return [str(server), '-m', str(models[0]), '--mmproj', str(models[1]), '--alias', 'local-reasoner',
            '--host', '127.0.0.1', '--port', str(port), '--ctx-size', str(c['context_size']),
            '--parallel', '1', '--fit', 'on', '--fit-target', str(c['fit_margin_mib']),
            '--threads', str(c['threads']), '--threads-batch', str(c['threads_batch']),
            '--batch-size', str(c['batch_size']), '--ubatch-size', str(c['ubatch_size']),
            '--flash-attn', c['flash_attention'], '--cache-type-k', c['cache_type'],
            '--cache-type-v', c['cache_type'], '--image-max-tokens', str(c['image_max_tokens']),
            '--reasoning-budget', str(c['reasoning_budget']), '--reasoning', 'on',
            '--reasoning-format', 'deepseek', '--jinja']


@contextlib.contextmanager
def serving(server, models, c, env, log):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = server_args(server, models, c, port)
    atomic(log.with_suffix('.command.json'), command)
    with log.open('a') as stream:
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'llama-server exited ({proc.returncode}); inspect {log}')
                try:
                    web_json(f'http://127.0.0.1:{port}/health', timeout=2)
                    break
                except (error.URLError, OSError, ValueError):
                    if time.monotonic() - start > c['startup_timeout_seconds']:
                        raise TimeoutError(f'Server readiness timeout; inspect {log}')
                    time.sleep(1)
            yield f'http://127.0.0.1:{port}'
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def payload(row, images, c):
    content = []
    for p in images:
        content += [{'type': 'text', 'text': p.parent.name},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(p.read_bytes()).decode()}}]
    content.append({'type': 'text', 'text': 'Category: ' + row['metadata']['category'] +
                    '\nQuestion: ' + row['problem'] + '\nDataset answer: ' + tagged_text(row['solution'], 'answer')})
    return {'model': 'local-reasoner', 'messages': [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': content}], 'temperature': c['temperature'],
            'max_tokens': c['max_tokens'], 'seed': c['seed'], 'cache_prompt': True,
            'chat_template_kwargs': {'enable_thinking': True}, 'reasoning_format': 'deepseek'}


def parse_response(result, *, allow_conflict=False):
    choices = result.get('choices')
    if not isinstance(choices, list) or not choices:
        raise ValueError('Missing completion choice')
    choice = choices[0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('Truncated/non-stop response is not training data')
    reasoning = choice['message'].get('reasoning_content')
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError('Thinking trace is missing; refusing a non-thinking training target')
    content = choice['message'].get('content')
    if not isinstance(content, str):
        raise ValueError('Missing final JSON content')
    # Preserve internal reasoning separately; only final, short explanation is a training target.
    text = content.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    data = json.loads(text)
    if set(data) != {'explanation', 'evidence_status'}:
        raise ValueError('Unexpected explanation schema')
    explanation = data['explanation']
    if not isinstance(explanation, str) or not explanation.strip() or len(explanation.split()) > 180:
        raise ValueError('Explanation must contain 1..180 words')
    if re.search(r'</?(?:think|answer)\b', explanation, re.I):
        raise ValueError('Nested training tags are forbidden')
    if data['evidence_status'] not in ('grounded', 'limited', 'conflict'):
        raise ValueError('Unknown evidence status')
    if data['evidence_status'] == 'conflict' and not allow_conflict:
        raise ValueError('Model flagged label conflict; manual review required')
    return '<think>' + explanation.strip() + '</think>', data['evidence_status']


def image_inventory(rows, indices, image_root):
    paths, hashes = {}, {}
    for i in indices:
        images = [image_path(p, image_root) for p in rows[i]['image']]
        if len(images) != 6 or {p.parent.name for p in images} != set(CAMERAS):
            raise ValueError(f'{i}: six distinct cameras required')
        for p in images:
            if not p.is_relative_to(image_root.resolve()):
                raise ValueError('Image escapes image root')
            if str(p) not in hashes:
                if p.read_bytes()[:3] != b'\xff\xd8\xff':
                    raise ValueError('Expected a JPEG')
                hashes[str(p)] = file_sha(p)
        paths[i] = images
    return paths, hashes


def read_journal(path, signature):
    latest = {}
    if not path.exists():
        return latest
    with path.open('rb+') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                stream.truncate(start)  # only an interrupted last write; earlier complete lines stay intact
                break
            record = json.loads(line)
            if record['signature'] != signature:
                raise ValueError('Journal identity changed')
            latest[record['id']] = record
    return latest


def publish(out, rows, manifest, signature, latest):
    indices = manifest['indices']['train']
    records = [latest[i] for i in indices if i in latest]
    complete = len(records) == len(indices) and all(x['status'] == 'ok' for x in records)
    data = {'schema_version': 1, 'kind': 'train_reasoning_overlay', 'dataset_sha256': dataset_hash(rows),
            'split_manifest': manifest, 'complete': complete, 'generation_manifest': str(out / 'generation_manifest.json'),
            'generation_signature': signature, 'semantic_review': 'automated_format_and_self_report_only; not independently verified',
            'records': records}
    atomic(out / 'reasoning_overlay.json', data)
    summary = {'complete': complete, 'total_train': len(indices), 'ok': sum(x['status']=='ok' for x in records),
               'failed': sum(x['status']!='ok' for x in records), 'pending': len(indices)-len(records)}
    atomic(out / 'status.json', summary)
    return summary


def tune(server, models, env, c, rows, paths, indices, out, signature):
    saved = out / 'tuning.json'
    if saved.exists():
        data = json.loads(saved.read_text())
        if data['signature'] != signature:
            raise ValueError('Tuning inputs changed')
        return data['selected_config']
    # Stratify exclusively by TRAIN category. Do not select settings against test answers.
    rng = random.Random(c['seed'])
    buckets = {}
    for i in indices:
        buckets.setdefault(rows[i]['metadata']['category'], []).append(i)
    for values in buckets.values():
        rng.shuffle(values)
    chosen = []
    while len(chosen) < min(c['benchmark_samples'],len(indices)):
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < c['benchmark_samples']:
                chosen.append(buckets[key].pop())
    chosen.sort(key=lambda i:(rows[i]['metadata']['frame_id'],i))
    candidates = [dict(c, threads=t) for t in sorted({4,6,8,12,c['threads']})]
    # Record quality-changing alternatives but never automatically lower image resolution.
    candidates += [dict(c, cache_type='f16'), dict(c, image_max_tokens=512)]
    progress = out/'tuning_progress.json'
    cached = json.loads(progress.read_text()) if progress.exists() else []
    results = []
    for number, cfg in enumerate(candidates):
        item = {'config': cfg, 'samples': [], 'eligible': cfg['image_max_tokens']==c['image_max_tokens'] and cfg['cache_type']==c['cache_type']}
        previous = next((x for x in cached if x['config']==cfg and [a['id'] for a in x['samples']]==chosen and 'error' not in x), None)
        if previous is not None:
            item['samples'] = previous['samples']
            print(f'기존 속도 표본 재사용 {number+1}/{len(candidates)}', flush=True)
            results.append(item)
            continue
        print(f'속도 표본 {number+1}/{len(candidates)}: threads={cfg["threads"]}, cache={cfg["cache_type"]}, image={cfg["image_max_tokens"]}', flush=True)
        try:
            with serving(server,models,cfg,env,out/f'tune_{number}.log') as endpoint:
                # Warm up without counting compilation/cold startup as steady-state generation time.
                web_json(endpoint+'/v1/chat/completions',payload(rows[chosen[0]],paths[chosen[0]],cfg),cfg['timeout_seconds'])
                for i in chosen:
                    start=time.monotonic()
                    raw=web_json(endpoint+'/v1/chat/completions',payload(rows[i],paths[i],cfg),cfg['timeout_seconds'])
                    elapsed=time.monotonic()-start
                    try:
                        parse_response(raw, allow_conflict=True);valid=True
                    except (ValueError,KeyError,TypeError):
                        valid=False
                    item['samples'].append({'id':i,'seconds':elapsed,'format_valid':valid,'response':raw})
        except (RuntimeError,OSError,ValueError,TimeoutError) as exc:
            item['error']=str(exc)
        item['valid_count']=sum(x['format_valid'] for x in item['samples'])
        item['seconds']=sum(x['seconds'] for x in item['samples'])
        results.append(item)
        atomic(out/'tuning_progress.json',results)
    return select_tuning(results, chosen, c, out, signature)


def select_tuning(results, chosen, c, out, signature):
    for item in results:
        item['eligible'] = item['config']['image_max_tokens']==c['image_max_tokens'] and item['config']['cache_type']==c['cache_type']
        for sample in item['samples']:
            try:
                _, evidence = parse_response(sample['response'], allow_conflict=True)
                sample.update(format_valid=True, evidence_status=evidence, training_eligible=evidence!='conflict')
                sample.pop('validation_error', None)
            except (ValueError, KeyError, TypeError) as exc:
                sample.update(format_valid=False, training_eligible=False, validation_error=str(exc))
        item['valid_count'] = sum(x['format_valid'] for x in item['samples'])
        item['conflict_count'] = sum(x.get('evidence_status')=='conflict' for x in item['samples'])
        item['seconds'] = sum(x['seconds'] for x in item['samples'])
    atomic(out/'tuning_progress.json', results)
    eligible=[r for r in results if r['eligible'] and [x['id'] for x in r['samples']]==chosen and r['valid_count']==len(chosen) and 'error' not in r]
    if not eligible:
        raise RuntimeError('No candidate completed every benchmark row with valid output; inspect tuning_progress.json. No quality fallback was selected.')
    best=min(eligible,key=lambda x:x['seconds'])
    atomic(out/'tuning.json',{'signature':signature,'selection':'minimum end-to-end time; same cache/image budgets; all sampled outputs format-valid, NOT semantic-quality validation',
                  'selected_config':best['config'],'train_sample_ids':chosen,'candidates':results})
    return best['config']


def repair_tuning_manifest(out, identity, signature):
    plan = out/'generation_manifest.json'
    old = json.loads(plan.read_text())
    body = {k:v for k,v in old.items() if k!='signature'}
    if dataset_hash([body]) != old['signature']:
        raise ValueError('Saved manifest checksum mismatch')
    if any((out/name).exists() for name in ('events.jsonl','reasoning_overlay.json','tuning.json')):
        raise ValueError('Repair is only allowed before production generation and before tuning selection')
    if not (out/'tuning_progress.json').is_file():
        raise ValueError('No saved benchmark to repair')
    if {k:v for k,v in body.items() if k!='generator_sha256'} != {k:v for k,v in identity.items() if k!='generator_sha256'}:
        raise ValueError('Inputs/model/prompt/settings changed; refusing benchmark reuse')
    backup = out/('tuning_repair_backup_'+old['signature'])
    backup.mkdir(exist_ok=True)
    for name in ('generation_manifest.json','tuning_progress.json'):
        target=backup/name
        if not target.exists():
            shutil.copyfile(out/name,target)
    atomic(plan,dict(identity,signature=signature))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/local_reasoning.json'))
    p.add_argument('--data',default=str(ROOT/'data/drivelm.json'))
    p.add_argument('--images',default=str(ROOT/'data/nuscenes'))
    p.add_argument('--split',default=str(ROOT/'outputs/ablation_study/20261102_controlled/split_manifest.json'))
    p.add_argument('--output',default=str(ROOT/'outputs/local_reasoning/qwen38_27b_iq3'))
    p.add_argument('--server-binary',help='Use an existing CUDA llama-server instead of downloading a release')
    p.add_argument('--dry-run',action='store_true',help='Validate source/split/images without downloads or generation')
    p.add_argument('--skip-tuning',action='store_true',help='Use configured settings; not a claim of optimal speed')
    p.add_argument('--limit',type=int,help='Generate at most N remaining train rows; partial overlay is not training-ready')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--repair-tuning-only',action='store_true',help='Revalidate saved benchmark before any production generation; never launch generation')
    args=p.parse_args()
    c=validate_config(json.loads(Path(args.config).read_text()))
    if args.limit is not None and args.limit<1:
        p.error('--limit must be positive')
    rows=load_rows(args.data);manifest=json.loads(Path(args.split).read_text())
    load_split(rows,args.split,manifest['seed'],manifest['val_size'],manifest['test_size'],create=False)
    indices=manifest['indices']['train']
    paths,hashes=image_inventory(rows,indices,Path(args.images))
    if args.dry_run:
        print(json.dumps({'train':len(indices),'unique_images':len(hashes),'config':c,'downloads':False,'original_data_untouched':True},ensure_ascii=False,indent=2));return
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        server,models,env,runtime=bootstrap(c,out,args.server_binary)
        identity={'dataset_sha256':dataset_hash(rows),'split_manifest':manifest,'images':hashes,'config':c,
                  'system_prompt':SYSTEM,'runtime':runtime,'skip_tuning':args.skip_tuning,
                  'generator_sha256':file_sha(Path(__file__)),
                  'helper_hashes':{name:file_sha(ROOT/'scripts'/name) for name in ('data_utils.py','generate_reasoning.py','reasoning_source.py')}}
        signature=dataset_hash([identity]);plan=out/'generation_manifest.json'
        if args.repair_tuning_only:
            repair_tuning_manifest(out, identity, signature)
        if plan.exists():
            if json.loads(plan.read_text())['signature']!=signature:
                raise ValueError('Generation inputs/config/code changed; use a new output directory')
        else:
            atomic(plan,dict(identity,signature=signature))
        selected=c if args.skip_tuning else tune(server,models,env,c,rows,paths,indices,out,signature)
        if args.repair_tuning_only:
            print('속도 측정 복구 완료. 전체 생성은 기존 실행 명령으로 시작하세요.');return
        journal=out/'events.jsonl';latest=read_journal(journal,signature)
        if any(i not in set(indices) for i in latest):
            raise ValueError('Journal includes non-train row')
        remaining=[i for i in indices if i not in latest or (args.retry_failed and latest[i]['status']!='ok')]
        remaining.sort(key=lambda i:(rows[i]['metadata']['scene_id'],rows[i]['metadata']['frame_id'],i))
        if args.limit:
            remaining=remaining[:args.limit]
        if remaining:
            with serving(server,models,selected,env,out/'generation.log') as endpoint:
                for i in remaining:
                    event={'id':i,'signature':signature,'source_row_sha256':dataset_hash([rows[i]]),'status':'failed'}
                    for attempt in range(selected['attempts']):
                        start=time.monotonic()
                        try:
                            raw=web_json(endpoint+'/v1/chat/completions',payload(rows[i],paths[i],selected),selected['timeout_seconds'])
                            event['raw_response']=raw
                            thinking,evidence=parse_response(raw)
                            event.update(status='ok',thinking=thinking,evidence_status=evidence)
                            event.pop('error',None)
                            break
                        except (OSError,ValueError,KeyError,TypeError,TimeoutError) as exc:
                            event['error']=str(exc)
                        finally:
                            event['last_attempt_seconds']=time.monotonic()-start
                            event['attempts']=attempt+1
                    append_event(journal,event);latest[i]=event
                    summary={'ok':sum(x['status']=='ok' for x in latest.values()),'failed':sum(x['status']!='ok' for x in latest.values())}
                    if len(latest)%10==0:
                        publish(out,rows,manifest,signature,latest)
                    print(f'QA {i}: {event["status"]}; 완료 {summary["ok"]}/{len(indices)}, 실패 {summary["failed"]}',flush=True)
        summary=publish(out,rows,manifest,signature,latest)
        print(json.dumps(summary,ensure_ascii=False));print('학습용 추론:',out/'reasoning_overlay.json')
        if summary['failed']:
            raise SystemExit('실패/충돌 문항이 있어 학습은 차단됩니다. events.jsonl을 검토한 뒤 --retry-failed 또는 새 출력 경로로 재생성하세요.')


if __name__=='__main__':
    main()
