#!/usr/bin/env python3
"""Publish compact result tables to reports/; keep weights and raw QA in ignored outputs/."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from analyze_ablation_results import read_rows, scored_rows, summarize

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    baseline=ROOT/'outputs/ablation_study/20261102_controlled'
    old_model=baseline/'models/reason_b12_response'
    old_response=baseline/'evaluations/reason_b12_response_test.json'
    old_metric=Path(str(old_response)+'.metrics.json')
    local_metric=run/'comparison/comparison.json'
    old=read(old_metric);new=read(local_metric)
    split=read(run/'split_manifest.json')
    if split!=read(old_model/'split_manifest.json'):
        raise ValueError('Comparison requires the same split')
    data=read_rows(ROOT/'data/drivelm.json')
    table=[];sources=[]
    for label,model,response,scores,counts in [
        ('original_reason_b12',old_model,old_response,old['graph_scores'],old['graph_counts']),
        ('local_reason_b12',run,run/'reason.json',new['rows'][0],new['counts']['reason'])]:
        responses=read(response)
        if len(responses)!=len(split['indices']['test']) or {int(r['id']) for r in responses}!=set(split['indices']['test']):
            raise ValueError('Response IDs must exactly match test split')
        fixed=summarize(scored_rows(data,split['indices']['test'],{r['id']:r for r in responses}))
        config=read(model/'training_config.json');done=read(model/'completed.json')
        skipped={r['index'] for r in read(model/'skipped_samples.json') if r['split']=='train'}
        row={'model':label, **{k:v for k,v in scores.items() if k!='model'},
             'accuracy_denominator':counts['accuracy'], 'behavior_correct':fixed['behavior']['correct'],
             'behavior_total':fixed['behavior']['n'],'stop_correct':fixed['stop']['correct'],'stop_total':fixed['stop']['n'],
             'fixed_tag0_correct':fixed['fixed_tag0']['correct'],'fixed_tag0_total':fixed['fixed_tag0']['n'],
             'valid_answers':counts['valid_answers'],'test_total':counts['total'],
             'train_count':len(split['indices']['train'])-len(skipped),'batch_size':config['batch_size'],
             'gradient_accumulation_steps':config['gradient_accumulation_steps'],
             'train_on_responses_only':config['train_on_responses_only'],'finished_epoch':done['epoch'],
             'selected_checkpoint':Path(done['best_model_checkpoint']).name}
        table.append(row)
        for path in (response,model/'training_config.json',model/'completed.json'):
            sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in (old_metric,local_metric,run/'split_manifest.json'):
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    a,b=table
    notes=['Graph Accuracy의 채점 대상은 모델별로 다르므로 동일 행동 문항과 고정 tag0 결과도 함께 본다.',
           f'원본/로컬 추론 실험의 train 문항 수는 각각 {a["train_count"]}/{b["train_count"]}이다. 추론 교체만의 효과로 단정할 수 없다.',
           '두 조건 모두 batch 12 × 누적 1, 응답만 학습, 최대 3 epochs이다. validation/test split은 동일하다.',
           f'원본 추론을 유지한 validation 전체 응답 loss로 체크포인트를 선택했다. 원본: {a["selected_checkpoint"]}, 로컬: {b["selected_checkpoint"]}.',
           'Match_noGPT와 Final_noGPT는 GPT 채점을 포함하는 논문 지표와 다르다. 단일 seed의 탐색적 비교다.',
           '원본 QA·생성 전문·이미지·모델 가중치는 포함하지 않는다. 출처 경로는 로컬 재현용이며 해당 파일은 Git에서 제외된다.']
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps({'rows':table,'notes':notes,'sources':sources},ensure_ascii=False,indent=2)+'\n')
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    a,b=table
    lines=['# 원본 추론과 Qwen 로컬 추론의 Reason 성능 비교','',
           '| 지표 | 원본 추론 | 로컬 추론 |','| --- | ---: | ---: |']
    for key in ('Accuracy','Bleu_1','Bleu_2','Bleu_3','Bleu_4','ROUGE_L','CIDEr','Match_noGPT','Final_noGPT'):
        lines.append(f'| {key} | {a[key]:.4f} | {b[key]:.4f} |')
    for title,n,d in [('행동 정답','behavior_correct','behavior_total'),('정지 문항 정답','stop_correct','stop_total'),('고정 tag0 정답','fixed_tag0_correct','fixed_tag0_total'),('정상 응답','valid_answers','test_total')]:
        lines.append(f'| {title} | {a[n]}/{a[d]} | {b[n]}/{b[d]} |')
    for title,key in [('Accuracy 채점 문항 수','accuracy_denominator'),('실제 train 문항 수','train_count'),('선택 체크포인트','selected_checkpoint')]:
        lines.append(f'| {title} | {a[key]} | {b[key]} |')
    lines+=['',f'동일 행동 {a["behavior_total"]}문항에서 원본 추론 모델은 {a["behavior_correct"]}개, 로컬 추론 모델은 {b["behavior_correct"]}를 맞혔다.','',*['- '+n for n in notes],'',
            '전체 수치는 [CSV](comparison.csv), 출처와 SHA256은 [summary.json](summary.json)에 기록했다.','']
    (out/'comparison.md').write_text('\n'.join(lines))
    print(out/'comparison.md')

if __name__=='__main__':main()
