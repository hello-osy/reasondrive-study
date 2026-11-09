#!/usr/bin/env python3
"""Export an explicit conflict-excluded train overlay without changing the dataset/split."""
import argparse
import hashlib
import json
from pathlib import Path
from data_utils import dataset_hash, load_split, write_json
from generate_reasoning import load_rows
from local_reasoning import parse_response
from reasoning_source import load_reasoning_source

ROOT = Path(__file__).resolve().parents[1]


def select(source, output, rows):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Do not overwrite the original generation artifact')
    data = json.loads(source.read_text())
    if data.get('kind') != 'train_reasoning_overlay' or data.get('schema_version') != 1 or data.get('selection_policy'):
        raise ValueError('Expected the original generation overlay')
    if data['dataset_sha256'] != dataset_hash(rows):
        raise ValueError('Source dataset changed')
    manifest=data['split_manifest']
    train=set(manifest['indices']['train']);seen=set();accepted=[];excluded=[]
    for record in data['records']:
        index=record['id']
        if type(index) is not int or index not in train or index in seen:
            raise ValueError('Unexpected/duplicate train ID')
        seen.add(index)
        if record['source_row_sha256'] != dataset_hash([rows[index]]):
            raise ValueError('Source row changed')
        thinking,evidence=parse_response(record['raw_response'],allow_conflict=True)
        if record['status']=='ok' and evidence in ('grounded','limited'):
            if record.get('thinking') != thinking:
                raise ValueError('Stored reasoning differs from generated response')
            accepted.append(record)
        elif record['status']=='failed' and evidence=='conflict':
            excluded.append({'id':index,'source_row_sha256':record['source_row_sha256'],'evidence_status':'conflict'})
        else:
            raise ValueError('Only explicit conflicts may be excluded; other failures require repair')
    if seen != train:
        raise ValueError('Generation is unfinished; pending rows must not be silently dropped')
    result=dict(data,records=accepted,complete=True,selection_policy='exclude_conflicts',excluded_records=excluded,
                selection_source={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    if output.exists():
        if json.loads(output.read_text())!=result:
            raise ValueError('Output exists with different data; use a new output filename')
    else:
        write_json(output,result)
    load_reasoning_source(output,rows,manifest)
    print(f'train={len(accepted)}, excluded_conflict={len(excluded)}, validation/test unchanged: {output}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(ROOT/'outputs/local_reasoning/qwen38_27b_iq3/reasoning_overlay.json'))
    p.add_argument('--output',default=str(ROOT/'outputs/local_reasoning/qwen38_27b_iq3/reasoning_success_only.json'))
    p.add_argument('--data',default=str(ROOT/'data/drivelm.json'))
    a=p.parse_args();rows=load_rows(a.data)
    manifest=json.loads(Path(a.source).read_text())['split_manifest']
    # Validate the full scene partition before exporting an explicitly reduced training subset.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        split=Path(tmp)/'split.json';write_json(split,manifest)
        load_split(rows,split,manifest['seed'],manifest['val_size'],manifest['test_size'],create=False)
    select(a.source,a.output,rows)


if __name__=='__main__':main()
