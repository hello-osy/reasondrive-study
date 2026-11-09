"""Frozen sequential ablation study: prompt diagnostics, training, validation, final test.
No API/GPT calls. Never edits old runs; restart the same command to resume.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from data_utils import write_json, dataset_hash
from experiment_config import load_experiment, validate_config, config_hash, backend_settings
from evaluate import read_data

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/report_experiments/modified/20261028_180046_j4kzqit2'

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_modified_baseline(model_dir):
    """Derive new candidates from the user's completed modified run, not defaults."""
    model_dir = Path(model_dir)
    saved = json.loads((model_dir/'training_config.json').read_text())
    completed = json.loads((model_dir/'completed.json').read_text())
    if completed.get('smoke', True) or saved.get('mode') != 'reason':
        raise ValueError('The baseline must be a completed non-smoke modified Reason run')
    base = validate_config(copy.deepcopy(saved['experiment']))
    if config_hash(base) != saved.get('experiment_sha256'):
        raise ValueError('Saved baseline experiment fingerprint differs')
    for key, value in backend_settings(base, 'cuda').items():
        if saved.get(key) != value:
            raise ValueError(f'Saved baseline effective setting differs: {key}')
    return base

def prepare(run):
    if (run/'plan.json').exists():return json.loads((run/'plan.json').read_text())
    if run.exists() and any(run.iterdir()):raise ValueError('New run directory must be empty')
    base=load_modified_baseline(OLD/'reason')
    run.mkdir(parents=True,exist_ok=True)
    snapshot=run/'source_snapshot';snapshot.mkdir()
    shutil.copytree(ROOT/'scripts',snapshot/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'configs',snapshot/'configs')
    shutil.copytree(ROOT/'vendor/public_reasondrive',snapshot/'vendor/public_reasondrive')
    for name in ['split_manifest.json','graph_context.json']:
        shutil.copy2(OLD/name,run/name)
    variants=[]
    # One-factor comparisons in Reason: batch12/30 × full/response loss.
    # Simple controls use the same batch12 and each loss policy.
    for name,mode,batch,response in [('reason_b12_full','reason',12,False),
                                   ('simple_b12_full','simple',12,False),
                                   ('reason_b12_response','reason',12,True),
                                   ('simple_b12_response','simple',12,True),
                                   ('reason_b30_response','reason',30,True)]:
        config=copy.deepcopy(base);config['experiment_profile']='ablation'
        config['description']=name+'; fixed scene split; three epochs; seed3407'
        config['common'].update(batch_size=batch,gradient_accumulation_steps=1,
                                train_on_responses_only=response,finetune_attention_modules=False)
        validate_config(config);path=run/'configs'/f'{name}.json';write_json(path,config)
        variants.append({'name':name,'mode':mode,'config':str(path),'config_sha256':sha(path)})
    data=ROOT/'data/drivelm.json'
    plan={'schema_version':1,'created':now(),'root':str(ROOT),'run':str(run),
          'data':str(data),'data_file_sha256':sha(data),'image_root':str(ROOT/'data/nuscenes'),
          'split_sha256':sha(run/'split_manifest.json'),'context_sha256':sha(run/'graph_context.json'),
          'source_sha256':{str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()},
          'legacy_reason':str(OLD/'reason'),'legacy_simple':str(OLD/'simple'),'variants':variants,
          'baseline_reference':{'profile':'modified','run':str(OLD),
              'training_config_sha256':sha(OLD/'reason/training_config.json'),
              'completed_sha256':sha(OLD/'reason/completed.json'),
              'experiment_sha256':config_hash(base),
              'meaning':'Candidates change selected settings of the final modified Reason recipe. Fresh training starts from the same pinned base weights; prompt diagnostics reuse the completed modified Reason adapter.'},
          'prompt_diagnostics':[{'name':'current_greedy','prompt':'current','sample':False},
                                {'name':'public_greedy','prompt':'public','sample':False},
                                {'name':'current_sample','prompt':'current','sample':True},
                                {'name':'public_sample','prompt':'public','sample':True}],
          'selection':'All training configurations frozen before any new validation. Validation tag0 exact accuracy selects a Reason candidate; test is never used to add/tune candidates.',
          'test_policy':'Evaluate all five preregistered new models once after every validation; no test-based retuning in this study.',
          'reference':{'paper':'https://arxiv.org/html/2504.10757v1#S4','Reason_Accuracy':.47,'Simple_Accuracy':.32,
                       'limitation':'Exact paper test IDs unavailable. Scene-disjoint results are not an exact paper benchmark reproduction. Final_noGPT differs from paper GPT-inclusive Final.'}}
    (run/'requirements.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())))
    write_json(run/'plan.json',plan)
    (run/'code.diff').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    report(run,plan);return plan

def metric(run,name,split):
    p=run/'evaluations'/f'{name}_{split}.json.metrics.json'
    return json.loads(p.read_text()) if p.exists() else None

def report(run,plan,write_selection=True):
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'state':'prepared'}
    lines=['# ReasonDrive ablation study','',f"상태: {status['state']} / 단계: {status.get('stage','준비')}",'',
      '개선의 기준은 사용자의 최종 수정본 실험이다. 이 수정본에서 Reason의 낮은 행동 예측 성능이 어떤 변경으로 개선되는지 비교한다. 공개 지시문 비교도 수정본 Reason 가중치를 사용하며, 공개 코드로 학습한 모델을 기준 모델로 삼지 않는다.','',
      '논문 기준 Accuracy: Simple 0.32, Reason 0.47. 논문의 정확한 평가 문항은 공개되지 않아, 본 실험은 고정된 장면 분리 평가에서 논문 수치 수준에 도달하는지 확인한다. Final_noGPT는 논문의 Final Score와 다르다.','',
      '고정 조건: 장면 분할 학습 4,218 / 검증 534 / 최종 평가 528 QA, seed 3407, 3 epoch, LR 2e-4, warmup 50, MLP LoRA r8/alpha8, 224px 6방향 입력. 새 학습은 동일한 초기 가중치에서 시작한다. 배치 12는 microbatch 12×1이며 논문의 1×12와 실행 방식은 다르다.','',
      '검증에서는 tag 0인 모든 문항을 고정 분모로 정확 일치 채점한다. 그래프에 따라 달라지는 논문식 Accuracy와는 다른 진단 지표다. 모든 후보를 검증한 뒤 사전 지정한 후보들의 최종 test를 한 번 평가한다. 기존 test 결과는 이미 열람한 자료이며 완전히 새로운 외부 검증셋은 아니다.','',
      '| 조건 | 검증 고정분모 정답률 | test Accuracy | test Final_noGPT | 유효응답 |',
      '| --- | --- | --- | --- | --- |']
    for v in plan['variants']:
        val=metric(run,v['name'],'validation');test=metric(run,v['name'],'test')
        va=val['fixed_tag0'] if val else None;gs=test.get('graph_scores',{}) if test else {}
        lines.append(f"| {v['name']} | {str(va['correct'])+'/'+str(va['n']) if va else '대기'} | {gs.get('Accuracy','대기')} | {gs.get('Final_noGPT','대기')} | {str(test['valid'])+'/'+str(test['total']) if test else '대기'} |")
    lines+=['','| 기존 Reason 동일 가중치 추론 비교 | 검증 고정분모 정답 수 |','| --- | --- |']
    for v in plan['prompt_diagnostics']:
        val=metric(run,'legacy_'+v['name'],'validation');a=val['fixed_tag0'] if val else None
        lines.append(f"| {v['name']} | {str(a['correct'])+'/'+str(a['n']) if a else '대기'} |")
    vals={v['name']:metric(run,v['name'],'validation') for v in plan['variants'] if v['mode']=='reason'}
    if vals and all(vals.values()):
        chosen=max(vals,key=lambda n:vals[n]['fixed_tag0']['accuracy'])
        lines+=['',f'검증 점수로 선정한 Reason 후보: `{chosen}` (동률이면 사전 실행 순서 우선).']
        if write_selection:
            selection = {'selected_reason':chosen,'criterion':'validation fixed_tag0 exact accuracy'}
            path = run/'selection.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if any(saved.get(key) != value for key, value in selection.items()):
                    raise ValueError('Recorded validation selection changed; inspect the underlying metrics')
            else:
                write_json(path,{**selection,'created':now()})
    if status['state']=='complete':
        lines+=['','## 단일 요인 비교','',
          '배치 효과는 Reason의 batch12/full과 기존 batch30/full, 그리고 batch12/response와 batch30/response를 비교한다. 손실 범위 효과는 동일 batch에서 full과 response를 비교한다. 배치를 바꾸면 업데이트 수와 warmup 비율도 함께 바뀌므로 배치 자체만의 영향으로 단정하지 않는다.','']
        for a,b in [('reason_b12_full','reason_b12_response'),('reason_b30_response','reason_b12_response'),('simple_b12_full','reason_b12_full'),('simple_b12_response','reason_b12_response')]:
            x=metric(run,a,'test');y=metric(run,b,'test')
            ax=x['fixed_tag0'];ay=y['fixed_tag0']
            delta=ay['accuracy']-ax['accuracy']
            lines.append(f'- {a} → {b}: 동일 tag0 전체 문항 정답률 {ax["correct"]}/{ax["n"]} → {ay["correct"]}/{ay["n"]} (차이 {delta*100:+.2f}%p).')
        best=metric(run,chosen,'test')['graph_scores']['Accuracy']
        lines+=['',f'검증에서 고른 Reason의 test Accuracy: {best:.4f}. '+('논문 수치 0.47 수준에 도달했다. 다만 평가 분할이 달라 정확한 논문 성능 재현으로 단정할 수 없다.' if best>=.47 else '논문 수치 0.47에 도달하지 못했다. 개선이 있더라도 논문 성능이 재현되었다고 보고하지 않는다.'),
          '', '이 결과는 seed 1개인 탐색적 ablation이다. 원인 확정과 일반화에는 반복 seed 및 새로운 평가 자료가 필요하다.']
    lines+=['','계획과 원시 결과: `plan.json`, `evaluations/`, `models/`, `logs/`. 원인 후보 감사: `../audit_current/audit.md`.']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')


def archive_uncheckpointed_attempt(run, model, variant):
    """Preserve an interrupted first epoch before restarting the same frozen config."""
    run, model = Path(run).resolve(), Path(model)
    if model.is_symlink() or model.resolve().parent != run/'models':
        raise ValueError('Only this study\'s direct model directories may be archived')
    if not model.exists() or not any(model.iterdir()):
        return None
    # Never move a complete model or any directory containing a checkpoint,
    # including an incomplete checkpoint that should fail explicit resume checks.
    if (model/'completed.json').exists() or any(model.glob('checkpoint-*')):
        return None
    saved_path = model/'training_config.json'
    if not saved_path.is_file():
        raise ValueError('Nonempty output has no training provenance; refusing automatic restart')
    saved = json.loads(saved_path.read_text())
    expected = load_experiment(variant['config'])
    if saved.get('experiment_sha256') != config_hash(expected) or saved.get('mode') != variant['mode']:
        raise ValueError('Interrupted model settings differ from the frozen variant')
    archive = run/'failed_attempts'/(model.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8])
    archive.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(model),str(archive))
    write_json(archive/'restart_record.json',{
        'archived_at':now(),'original_output':str(model),'reason':'Interrupted before the first saved checkpoint',
        'frozen_config':str(variant['config']),'frozen_config_sha256':sha(variant['config']),
        'restart_policy':'Restart from pinned base weights with the same frozen config; no checkpoint exists to resume.',
    })
    return archive


def execute(run,plan):
    lock=(run/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if sha(plan['data'])!=plan['data_file_sha256'] or sha(run/'split_manifest.json')!=plan['split_sha256'] or sha(run/'graph_context.json')!=plan['context_sha256']:raise ValueError('Frozen inputs changed')
    snap=run/'source_snapshot'
    for name,digest in plan['source_sha256'].items():
        if sha(snap/name)!=digest:raise ValueError(f'Frozen code changed: {name}')
    for v in plan['variants']:
        if sha(v['config'])!=v['config_sha256']:raise ValueError('Frozen configuration changed')
    env=os.environ.copy();env.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),PYTHONUNBUFFERED='1')
    script=snap/'scripts';(run/'logs').mkdir(exist_ok=True)
    state={'state':'running','pid':os.getpid(),'started':now()}
    def command(stage,argv):
        state.update(stage=stage,updated=now());write_json(run/'status.json',state);report(run,plan)
        print(now(),stage,flush=True)
        with (run/'logs'/f'{stage}.log').open('a') as log:
            subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def infer(name,model,split,prompt='current',sample=False):
        output=run/'evaluations'/f'{name}_{split}.json'
        argv=[script/'ablation_infer.py','--model_dir',model,'--data_path',plan['data'],'--image_root',plan['image_root'],
              '--output',output,'--split',split,'--prompt',prompt]
        if sample:argv.append('--sample')
        command(name+'_'+split,argv)
    try:
        for v in plan['prompt_diagnostics']:
            infer('legacy_'+v['name'],plan['legacy_reason'],'validation',v['prompt'],v['sample'])
        for v in plan['variants']:
            model=run/'models'/v['name']
            if not (model/'completed.json').exists():
                argv=[script/'vlm-finetuning.py','--config',v['config'],'--data_path',plan['data'],
                  '--base_image_path',plan['image_root'],'--split_path',run/'split_manifest.json',
                  '--output_dir',model,'--exact_output_dir','--mode',v['mode'],'--no-use_wandb']
                checkpoints=sorted(model.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
                if checkpoints:argv+=['--resume_from_checkpoint',checkpoints[-1]]
                else:
                    archive = archive_uncheckpointed_attempt(run, model, v)
                    if archive is not None:print('Archived interrupted attempt:',archive,flush=True)
                command(v['name']+'_train',argv)
            infer(v['name'],model,'validation')
        report(run,plan)
        if not (run/'selection.json').exists():raise ValueError('Validation selection missing before test')
        for v in plan['variants']:infer(v['name'],run/'models'/v['name'],'test')
        state.update(state='complete',stage='all_done',finished=now());write_json(run/'status.json',state);report(run,plan)
    except Exception as error:
        state.update(state='failed',error=str(error),updated=now());write_json(run/'status.json',state);report(run,plan);raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run_dir',required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare_only',action='store_true')
    mode.add_argument('--report_only',action='store_true',help='Regenerate REPORT.md from an existing plan without running jobs')
    args=p.parse_args();run=Path(args.run_dir).resolve()
    if args.report_only:
        plan=json.loads((run/'plan.json').read_text())
        report(run,plan,write_selection=False)
    else:
        plan=prepare(run)
        if not args.prepare_only:execute(run,plan)
    print(run,flush=True)
if __name__=='__main__':main()
