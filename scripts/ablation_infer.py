"""Resumable, provenance-checked ablation inference. Validation is used for selection."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from data_utils import (format_sample, clean_message_content, tagged_text, generated_answer,
                        dataset_hash, write_json, load_split)
from evaluate import read_data, route_and_score, prediction_text
from experiment_config import resolve_model_source


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def public_prompt(question, mode):
    result = ('You are an AI assistant for autonomous driving. Analyze the scene and reason through driving decisions carefully.\n\n'
              'Analyze the following driving scenario and provide reasoning:\n\n'
              + question + '\n\nPut your reasoning within <think></think> tags and your final answer within <answer></answer> tags.')
    if mode == 'simple':
        from public_baseline import simple_prompt
        result = simple_prompt(result)
    return result


def fixed_metrics(responses, rows):
    def eligible(r): return 0 in rows[int(r['id'])]['metadata']['original_qa_pair']['tag']
    groups = {'fixed_tag0': [r for r in responses if eligible(r)],
              'behavior': [r for r in responses if eligible(r) and rows[int(r['id'])]['metadata']['category'] == 'behavior']}
    groups['stop'] = [r for r in groups['behavior'] if 'the ego vehicle is not moving.' in r['gt_answer'].lower()]
    # Keep exact official equality; no normalization/answer repair of generated text.
    result = {'total': len(responses), 'valid': sum(r['status'] == 'ok' for r in responses),
              'length': sum(r.get('finish_reason') == 'length' for r in responses)}
    for name, selected in groups.items():
        n = len(selected); correct = sum(prediction_text(r) == r['gt_answer'] for r in selected)
        result[name] = {'correct': correct, 'n': n, 'accuracy': correct/n if n else None}
    return result


def validate_response_prefix(responses, expected, complete=False):
    """Validate row order, source references and scored/context identity before reuse."""
    if not isinstance(responses, list) or len(responses) > len(expected):
        raise ValueError('Invalid resume prefix length')
    if complete and len(responses) != len(expected):
        raise ValueError('Completed inference has an incomplete response set')
    for response, (identifier, row, scored) in zip(responses, expected):
        if (response.get('id') != str(identifier) or response.get('question') != row['problem']
                or response.get('gt_answer') != tagged_text(row['solution'], 'answer')
                or response.get('scored') is not scored):
            raise ValueError(f'Resumed response differs from source row {identifier}')
        if response.get('status') not in ('ok', 'invalid_format', 'error') or not isinstance(response.get('answer'), str):
            raise ValueError(f'Invalid response record: {identifier}')
        if response['status'] == 'ok':
            parsed = generated_answer(response['answer'])
            if response.get('final_answer') != parsed:
                raise ValueError(f'Resumed final answer differs from generated text: {identifier}')


def ensure_metadata(output, expected):
    """Never attach provenance to orphaned responses or completion artifacts."""
    output = Path(output)
    metafile = Path(str(output) + '.meta.json')
    artifacts = [output, *[Path(str(output) + suffix) for suffix in
                          ('.metrics.json', '.context.json', '.complete.json')]]
    if not metafile.exists():
        if any(path.exists() for path in artifacts):
            raise ValueError('Existing inference artifacts require their original metadata; refusing adoption')
        write_json(metafile, expected)
        return False
    saved = json.loads(metafile.read_text())
    if saved == expected:
        return False
    # The original frozen study recorded all other fields but not these two hashes.
    # Retain that evidence and require re-scoring; never infer original values for them.
    added = {'adapter_config_sha256', 'graph_context_sha256'}
    legacy_expected = {key: value for key, value in expected.items() if key not in added}
    if saved != legacy_expected:
        raise ValueError('Inference provenance changed')
    proof = Path(str(output) + '.complete.json')
    if proof.exists():
        raise ValueError('Legacy metadata cannot have a hardened completion proof')
    legacyfile = Path(str(output) + '.meta.legacy.json')
    if legacyfile.exists() and json.loads(legacyfile.read_text()) != saved:
        raise ValueError('Archived legacy metadata changed')
    write_json(legacyfile, saved)
    write_json(Path(str(output) + '.provenance_upgrade.json'), {
        'legacy_metadata_sha256': sha(legacyfile),
        'new_hashes_observed_at_upgrade': {key: expected[key] for key in sorted(added)},
        'limitation': 'Legacy output did not record adapter-config/context hashes; their historical values are unverified. Existing response IDs/references and old hashes are checked, and metrics are recomputed.',
    })
    write_json(metafile, expected)
    return True


def completion_proof(output, has_context):
    output = Path(output)
    paths = {'responses_sha256': output, 'metadata_sha256': Path(str(output) + '.meta.json'),
             'metrics_sha256': Path(str(output) + '.metrics.json')}
    if has_context:
        paths['context_responses_sha256'] = Path(str(output) + '.context.json')
    return {'schema_version': 1, **{key: sha(path) for key, path in paths.items()}}


def verified_complete(output, has_context):
    """A legacy metrics file alone is not completion proof; it will be re-scored."""
    proof = Path(str(output) + '.complete.json')
    if not proof.exists():
        return False
    try:
        actual = completion_proof(output, has_context)
    except FileNotFoundError as error:
        raise ValueError('Completed inference is missing a recorded artifact') from error
    if json.loads(proof.read_text()) != actual:
        raise ValueError('Completed inference artifacts changed')
    return True


def make_predictor(cfg, modeldir, args):
    """Load GPU libraries only when some response still needs generation."""
    from unsloth import FastVisionModel
    import torch
    from transformers import set_seed
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    source=cfg['model_source'];path,_=resolve_model_source(source['model_id'],source['resolved_revision'])
    model,processor=FastVisionModel.from_pretrained(path,load_in_4bit=True)
    model=PeftModel.from_pretrained(model,str(modeldir),is_trainable=False)
    FastVisionModel.for_inference(model)
    def predict(identifier,row,scored):
        record={'id':str(identifier),'question':row['problem'],'gt_answer':tagged_text(row['solution'],'answer'),
                'answer':'','final_answer':'','status':'error','scored':scored}
        sample=format_sample(row,args.image_root,cfg['mode'],training=False)
        sample['messages']=clean_message_content(sample['messages'])
        if args.prompt=='public':sample['messages'][0]['content'][-1]['text']=public_prompt(row['problem'],cfg['mode'])
        images,_=process_vision_info(sample['messages'])
        prompt=processor.apply_chat_template(sample['messages'],add_generation_prompt=True,tokenize=False)
        inputs=processor(images=images,text=prompt,add_special_tokens=False,return_tensors='pt').to('cuda')
        stable_seed=args.seed+int(hashlib.sha256(str(identifier).encode()).hexdigest()[:8],16)%1000000
        set_seed(stable_seed)
        generation=copy.deepcopy(model.generation_config)
        generation.max_length=20; generation.max_new_tokens=512; generation.do_sample=args.sample
        if args.sample:generation.temperature=.7;generation.top_p=.9
        else:generation.temperature=1.;generation.top_p=1.;generation.top_k=50
        with torch.inference_mode():ids=model.generate(**inputs,generation_config=generation)
        new=ids[:,inputs.input_ids.shape[1]:]
        record['answer']=processor.batch_decode(new,skip_special_tokens=True)[0].strip()
        eos=processor.tokenizer.eos_token_id
        record['generated_tokens']=int(new.shape[1])
        ended = bool(new.shape[1]) and int(new[0,-1]) in (eos if isinstance(eos,list) else [eos])
        record['finish_reason']='eos' if ended else ('length' if new.shape[1] >= 512 else 'other')
        try:record['final_answer']=generated_answer(record['answer']);record['status']='ok'
        except ValueError as e:record['status']='invalid_format';record['error']=str(e)
        return record
    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_dir', required=True); p.add_argument('--data_path', required=True)
    p.add_argument('--image_root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['validation', 'test'], required=True)
    p.add_argument('--prompt', choices=['current','public'], default='current')
    p.add_argument('--sample', action='store_true'); p.add_argument('--seed', type=int, default=3407)
    args=p.parse_args(); output=Path(args.output); modeldir=Path(args.model_dir)
    rows=read_data(args.data_path); cfg=json.loads((modeldir/'training_config.json').read_text())
    split=json.loads((modeldir/'split_manifest.json').read_text())
    load_split(rows, modeldir/'split_manifest.json', split['seed'], split['val_size'], split['test_size'])
    complete=json.loads((modeldir/'completed.json').read_text())
    if complete.get('smoke', True): raise ValueError('Full completed training required')
    indices=split['indices'][args.split]
    if args.split=='validation':
        indices=[i for i in indices if 0 in rows[i]['metadata']['original_qa_pair']['tag']]
    context = json.loads((modeldir/'graph_context.json').read_text()) if args.split == 'test' else None
    meta={'schema_version':1,'split':args.split,'indices':indices,'dataset_sha256':dataset_hash(rows),
          'adapter_sha256':sha(modeldir/'adapter_model.safetensors'), 'mode':cfg['mode'],
          'prompt':args.prompt,'do_sample':args.sample,'seed':args.seed,'budget':512,
          'model_source':cfg['model_source'], 'training_config_sha256':sha(modeldir/'training_config.json'),
          'selection_metric':'fixed_tag0 exact accuracy; no graph gating, all failures retained',
          'adapter_config_sha256':sha(modeldir/'adapter_config.json'),
          'graph_context_sha256':sha(modeldir/'graph_context.json') if context is not None else None}
    upgraded = ensure_metadata(output, meta)
    donefile=Path(str(output)+'.metrics.json');contextfile=Path(str(output)+'.context.json')
    has_completion = donefile.exists() or Path(str(output)+'.complete.json').exists()
    results=json.loads(output.read_text()) if output.exists() else []
    expected = [(i, rows[i], True) for i in indices]
    validate_response_prefix(results, expected, complete=has_completion)
    cr=[]
    if context is not None:
        if contextfile.exists():
            saved_context = json.loads(contextfile.read_text())
            if saved_context.get('graph_context') != context:
                raise ValueError('Resumed graph context differs from the model context')
            cr = saved_context['responses']
        validate_response_prefix(cr, [(item['id'], item, False) for item in context['items']], complete=has_completion)
    elif contextfile.exists():
        raise ValueError('Validation output must not contain test graph context')
    if not upgraded and verified_complete(output, context is not None):
        print('Already complete:',donefile,flush=True);return
    start=time.time()
    needs_generation = len(results) < len(indices) or (context is not None and len(cr) < len(context['items']))
    predict = make_predictor(cfg, modeldir, args) if needs_generation else None
    for i in indices[len(results):]:
        # Runtime/model failures stop the run; never manufacture empty answers for OOM.
        results.append(predict(i,rows[i],True))
        if len(results)%10==0:
            write_json(output,results);print(f'{args.split}: {len(results)}/{len(indices)} elapsed={time.time()-start:.1f}s',flush=True)
    write_json(output,results)
    metrics=fixed_metrics(results,rows)
    if args.split=='test':
        for item in context['items'][len(cr):]:
            cr.append(predict(item['id'],item,False));write_json(contextfile,{'graph_context':context,'responses':cr})
        write_json(contextfile,{'graph_context':context,'responses':cr})
        metrics['graph_scores'],metrics['graph_counts']=route_and_score(results,rows,context={'graph_context':context,'responses':cr})
    metrics['inference_seconds_this_attempt']=time.time()-start
    metrics['recomputed_without_generation'] = not needs_generation
    write_json(donefile,metrics)
    write_json(Path(str(output)+'.complete.json'), completion_proof(output, context is not None))
    print(json.dumps(metrics,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
