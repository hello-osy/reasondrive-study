"""CPU-only tensor probe of the installed vision collator; no model weights/forward."""
import os

# Set before importing torch, transformers or Unsloth Zoo. This probe is independent
# of the GPU study and must never create a CUDA context in its process.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['UNSLOTH_ZOO_DISABLE_GPU_INIT'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from data_utils import clean_message_content, format_sample, normalize_reasoning, write_json
from evaluate import read_data
from experiment_config import vision_collator_settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT/'outputs/report_experiments/modified/20261028_180046_j4kzqit2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(ids, needle, start=0):
    matches = [i for i in range(start, len(ids)-len(needle)+1) if ids[i:i+len(needle)] == needle]
    if len(matches) != 1:
        raise AssertionError(f'Expected one marker {needle}, found {len(matches)}')
    return matches[0]


def body_positions(tokenizer, ids, assistant_end, tag):
    # BPE may merge a tag delimiter with adjacent prose. Derive character spans
    # from a round-trip-verified encoding instead of searching standalone tag IDs.
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == ids, 'Offset mapping must match the actual collated token IDs'
    prefix = tokenizer.decode(ids[:assistant_end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    opening, closing = '<'+tag+'>', '</'+tag+'>'
    start = decoded.index(opening, len(prefix)) + len(opening)
    end = decoded.index(closing, start)
    assert end > start, f'{tag} body must be nonempty'
    # Include tokens that straddle body/tag boundaries; they still contain prose.
    positions = [i for i, (left, right) in enumerate(encoded['offset_mapping']) if left < end and right > start]
    assert positions, f'{tag} body has no token positions'
    return positions


def probe(args):
    import torch
    from transformers import AutoConfig, AutoProcessor
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator

    assert not torch.cuda.is_initialized(), 'The probe must remain CPU-only'
    source_run = Path(args.source_run)
    saved = json.loads((source_run/'reason/training_config.json').read_text())
    snapshot = Path(saved['model_source']['snapshot_path'])
    if not snapshot.is_dir():
        raise FileNotFoundError(f'Pinned local model snapshot is unavailable: {snapshot}')
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    tokenizer = processor.tokenizer
    # The installed collator needs config/media IDs plus an embedding dtype fallback.
    # This object has no parameters, weights, forward method, or device allocation.
    stub = SimpleNamespace(config=config, get_input_embeddings=lambda: SimpleNamespace(
        weight=torch.empty(0, dtype=torch.bfloat16)))
    rows = read_data(args.data)
    manifest = json.loads((source_run/'split_manifest.json').read_text())
    train = manifest['indices']['train']
    lengths = {i: len(tokenizer.encode(normalize_reasoning(rows[i]['thinking']), add_special_tokens=False))
               for i in train}
    ordered = sorted(train, key=lambda i: (lengths[i], i))
    chosen = list(dict.fromkeys([ordered[0], ordered[len(ordered)//2], ordered[-1], train[0]]))
    for i in train:
        if len(chosen) >= 4:
            break
        if i not in chosen:
            chosen.append(i)
    marker = lambda text: tokenizer.encode(text, add_special_tokens=False)
    assistant = marker('<|im_start|>assistant\n')
    image_id = config.image_token_id
    vision_end = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    summaries = []
    for mode in ('simple', 'reason'):
        samples = [clean_message_content(format_sample(rows[i], args.image_root, mode)['messages']) for i in chosen]
        samples = [{'messages': messages} for messages in samples]
        for padding in ('right', 'left'):
            tokenizer.padding_side = padding
            batches = {}
            for response_only in (False, True):
                settings = dict(saved, train_on_responses_only=response_only)
                collator = UnslothVisionDataCollator(stub, processor, **vision_collator_settings(settings))
                batch = collator(copy.deepcopy(samples))
                assert all(value.device.type == 'cpu' for value in batch.values() if torch.is_tensor(value))
                batches[response_only] = batch
            full, response = batches[False], batches[True]
            assert torch.equal(full['input_ids'], response['input_ids']), 'Loss policy must not change model inputs'
            assert torch.equal(full['attention_mask'], response['attention_mask'])
            assert (response['attention_mask'] == 0).any(), 'Different-length examples must exercise padding'
            item = {'mode': mode, 'padding_side': padding, 'batch_shape': list(response['input_ids'].shape), 'rows': []}
            for pos, index in enumerate(chosen):
                ids = response['input_ids'][pos].tolist()
                response_labels, full_labels = response['labels'][pos], full['labels'][pos]
                attention = response['attention_mask'][pos]
                assistant_start = locate(ids, assistant)
                assistant_end = assistant_start + len(assistant)
                image_mask = response['input_ids'][pos] == image_id
                pad_mask = attention == 0
                assert int(image_mask.sum()) == len(rows[index]['image']) * 64, 'Unexpected image expansion at 224px'
                assert torch.all(response_labels[:assistant_end] == -100), 'User/header tokens must be masked'
                assert torch.all(response_labels[image_mask] == -100), 'Image tokens must be masked'
                assert torch.all(response_labels[pad_mask] == -100), 'Padding tokens must be masked'
                assert torch.all(full_labels[image_mask | pad_mask] == -100), 'Full loss must still mask media/padding'
                assert (response_labels != -100).any(), 'Assistant response must have supervision'
                user_text_start = max(i for i, token in enumerate(ids[:assistant_start]) if token == vision_end) + 1
                full_user_count = int((full_labels[user_text_start:assistant_start] != -100).sum())
                assert full_user_count > 0, 'Full-text control must supervise user text'
                blocks = {}
                for name in (('think', 'answer') if mode == 'reason' else ('answer',)):
                    positions = body_positions(tokenizer, ids, assistant_end, name)
                    assert torch.all(response_labels[positions] != -100), f'{name} body was masked'
                    blocks[name] = {'body_tokens': len(positions), 'supervised_body_tokens': int((response_labels[positions] != -100).sum())}
                item['rows'].append({
                    'id': index, 'active_tokens': int(attention.sum()), 'image_tokens': int(image_mask.sum()),
                    'padding_tokens': int(pad_mask.sum()), 'full_supervised_tokens': int((full_labels != -100).sum()),
                    'response_supervised_tokens': int((response_labels != -100).sum()),
                    'full_user_text_supervised_tokens': full_user_count,
                    'response_user_and_header_supervised_tokens': int((response_labels[:assistant_end] != -100).sum()),
                    'response_image_supervised_tokens': int((response_labels[image_mask] != -100).sum()),
                    'response_padding_supervised_tokens': int((response_labels[pad_mask] != -100).sum()),
                    'blocks': blocks,
                })
            summaries.append(item)
    assert not torch.cuda.is_initialized(), 'CPU probe unexpectedly initialized CUDA'
    return {
        'schema_version': 1, 'status': 'passed', 'scope': 'Actual installed collator CPU tensors; config/dtype-only model stub; no weights, forward, backward or optimizer.',
        'script_sha256': sha(__file__), 'source_run': str(source_run.resolve()),
        'source_training_config_sha256': sha(source_run/'reason/training_config.json'),
        'data_sha256': sha(args.data), 'split_sha256': sha(source_run/'split_manifest.json'),
        'snapshot': str(snapshot), 'model_revision': saved['model_source']['resolved_revision'],
        'collator_file': inspect.getfile(UnslothVisionDataCollator),
        'collator_sha256': sha(inspect.getfile(UnslothVisionDataCollator)),
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'unsloth_zoo')},
        'cuda_initialized': torch.cuda.is_initialized(), 'device': 'cpu',
        'sample_selection': 'Shortest, median, longest normalized reasoning by tokenizer length, plus first training row; deduplicated and filled to four.',
        'selected_rows': [{'id': i, 'normalized_reasoning_tokens': lengths[i]} for i in chosen],
        'settings': {'max_seq_length': saved['max_seq_length'], 'image_size': saved['image_size'],
                     'instruction_part': '<|im_start|>user\n', 'response_part': '<|im_start|>assistant\n'},
        'batches': summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run', default=str(DEFAULT_RUN))
    parser.add_argument('--data', default=str(ROOT/'data/drivelm.json'))
    parser.add_argument('--image_root', default=str(ROOT/'data/nuscenes'))
    parser.add_argument('--output', default=str(ROOT/'outputs/ablation_study/audit_current/collator_probe.json'))
    args = parser.parse_args()
    result = probe(args)
    output = Path(args.output); write_json(output, result)
    lines = ['# 실제 collator 마스크 검증', '',
             '설치된 UnslothVisionDataCollator를 실제 CPU tensor로 실행했다. 학습 QA 4개를 선택해 Simple/Reason, 왼쪽/오른쪽 padding, full-text/response-only loss를 비교했다. 모든 검증을 통과했다.', '',
             '| 모드 | Padding | QA | 실제 입력 토큰 | Full 학습 토큰 | Response 학습 토큰 | 추론 본문 학습 토큰 | 정답 본문 학습 토큰 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for batch in result['batches']:
        for row in batch['rows']:
            lines.append(f"| {batch['mode']} | {batch['padding_side']} | {row['id']} | {row['active_tokens']} | {row['full_supervised_tokens']} | {row['response_supervised_tokens']} | {row['blocks'].get('think', {}).get('supervised_body_tokens', '—')} | {row['blocks']['answer']['supervised_body_tokens']} |")
    lines += ['', 'Response-only에서는 사용자·assistant header·이미지·padding 토큰의 학습 label이 모두 -100이었다. 각 QA의 response 학습 토큰은 0보다 컸으며, Reason의 think 본문과 모든 answer 본문은 마스킹되지 않았다. Full-text 대조군은 사용자 텍스트를 학습하고 이미지·padding은 마스킹했다. 두 손실 정책의 input_ids와 attention_mask는 동일했다.', '',
              '제한: 실제 모델 가중치 대신 로컬 고정 revision의 AutoConfig와 dtype만 제공하는 stub을 사용했다. Processor와 collator는 실제 설치본이다. CUDA context, 모델 forward/backward, optimizer update는 실행하지 않았다. 따라서 label 구성은 검증하지만 gradient·loss 계산·학습 성능은 검증하지 않는다. FastVisionModel 로더의 추가 tokenizer 수정 가능성도 이 CPU probe 범위 밖이다. 원시 token 수와 라이브러리/소스 hash는 동명의 JSON에 기록했다.']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'output':str(output),'cuda_initialized':result['cuda_initialized'],
                      'selected_rows':result['selected_rows'],'checked_batches':len(result['batches'])*2}, ensure_ascii=False))


if __name__ == '__main__':
    main()
