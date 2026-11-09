"""Local generator contract tests: no GPU/model downloads or quality claims."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import contextmanager
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data_utils import dataset_hash
from local_reasoning import parse_response, read_journal, publish, server_args, validate_config, main, CAMERAS, select_tuning, repair_tuning_manifest
from data_utils import load_split
from reasoning_source import load_reasoning_source, training_row


class LocalReasoningTests(unittest.TestCase):
    def test_final_only_and_reject_truncation_conflict_tags(self):
        response={'choices':[{'finish_reason':'stop','message':{'reasoning_content':'private internal draft',
                  'content':json.dumps({'explanation':'The vehicle ahead blocks the lane.', 'evidence_status':'grounded'})}}]}
        self.assertEqual(parse_response(response)[0], '<think>The vehicle ahead blocks the lane.</think>')
        missing=copy.deepcopy(response);missing['choices'][0]['message'].pop('reasoning_content')
        with self.assertRaises(ValueError):parse_response(missing)
        for status, explanation, finish in [('conflict','Contradiction','stop'),('grounded','<think>nested</think>','stop'),('limited','Cannot observe speed.','length')]:
            bad=copy.deepcopy(response);bad['choices'][0]['finish_reason']=finish
            bad['choices'][0]['message']['content']=json.dumps({'explanation':explanation,'evidence_status':status})
            with self.assertRaises(ValueError):parse_response(bad)

    def test_overlay_preserves_source_and_rejects_wrong_join(self):
        rows=[{'thinking':'<think>old</think>','problem':'Q','solution':'A','image':['image']}, {'thinking':'old2'}]
        manifest={'indices':{'train':[0],'validation':[1],'test':[]}}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);event={'id':0,'status':'ok','thinking':'<think>new</think>','source_row_sha256':dataset_hash([rows[0]])}
            summary=publish(out, rows, manifest, 'sig', {0:event})
            self.assertTrue(summary['complete'])
            mapping,_=load_reasoning_source(out/'reasoning_overlay.json', rows, manifest)
            changed=training_row(rows[0],0,'train',mapping)
            self.assertEqual(changed['thinking'],'<think>new</think>')
            self.assertEqual(rows[0]['thinking'],'<think>old</think>')
            for key in ('problem','solution','image'):self.assertEqual(changed[key],rows[0][key])
            self.assertEqual(training_row(rows[0],0,'validation',mapping),rows[0])
            original=json.loads((out/'reasoning_overlay.json').read_text())
            for mutation in ('missing','duplicate','test','hash','failed','split'):
                data=copy.deepcopy(original)
                if mutation=='missing':data['records']=[]
                if mutation=='duplicate':data['records']*=2
                if mutation=='test':data['records'][0]['id']=1
                if mutation=='hash':data['records'][0]['source_row_sha256']='wrong'
                if mutation=='failed':data['records'][0]['status']='failed'
                if mutation=='split':data['split_manifest']['indices']['train']=[1]
                (out/'reasoning_overlay.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError):load_reasoning_source(out/'reasoning_overlay.json',rows,manifest)

    def test_journal_recovers_only_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.jsonl';line=json.dumps({'id':2,'signature':'sig','status':'ok'})+'\n'
            p.write_text(line+'{"partial":')
            self.assertEqual(read_journal(p,'sig')[2]['status'],'ok')
            self.assertEqual(p.read_text(),line)
            with self.assertRaises(ValueError):read_journal(p,'other')

    def test_conflict_can_time_benchmark_but_never_enters_training(self):
        raw={'choices':[{'finish_reason':'stop','message':{'reasoning_content':'Checked views.',
             'content':json.dumps({'explanation':'The label may conflict.', 'evidence_status':'conflict'})}}]}
        with self.assertRaises(ValueError):parse_response(raw)
        self.assertEqual(parse_response(raw,allow_conflict=True)[1],'conflict')
        c={'image_max_tokens':1024,'cache_type':'q8_0','threads':8}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            result=[{'config':c,'samples':[{'id':590,'seconds':2.0,'response':raw}]}]
            self.assertEqual(select_tuning(result,[590],c,out,'signature'),c)
            self.assertEqual(result[0]['conflict_count'],1)
            self.assertFalse(result[0]['samples'][0]['training_eligible'])
            raw['choices'][0]['finish_reason']='length'
            with self.assertRaises(RuntimeError):select_tuning(result,[590],c,out,'signature')

    def test_repair_rejects_changed_inputs_or_started_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);old={'generator_sha256':'old','config':{'thinking':True}}
            (out/'generation_manifest.json').write_text(json.dumps(dict(old,signature=dataset_hash([old]))))
            (out/'tuning_progress.json').write_text('[]')
            new=dict(old,generator_sha256='new')
            with self.assertRaises(ValueError):repair_tuning_manifest(out,dict(new,config={}), 'sig')
            (out/'events.jsonl').write_text('')
            with self.assertRaises(ValueError):repair_tuning_manifest(out,new,dataset_hash([new]))
            (out/'events.jsonl').unlink()
            repair_tuning_manifest(out,new,dataset_hash([new]))
            self.assertEqual(json.loads((out/'generation_manifest.json').read_text())['generator_sha256'],'new')
            self.assertEqual(len(list(out.glob('tuning_repair_backup_*'))),1)

    def test_explicit_subset_excludes_only_conflicts_and_covers_train(self):
        rows=[{'thinking':'old'}, {'thinking':'old2'}, {'thinking':'validation'}]
        manifest={'indices':{'train':[0,1],'validation':[2],'test':[]}}
        event={'id':0,'status':'ok','thinking':'<think>new</think>','source_row_sha256':dataset_hash([rows[0]])}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);publish(out,rows,manifest,'sig',{0:event})
            p=out/'reasoning_overlay.json';data=json.loads(p.read_text())
            data.update(complete=True,selection_policy='exclude_conflicts',excluded_records=[{
                'id':1,'source_row_sha256':dataset_hash([rows[1]]),'evidence_status':'conflict'}])
            p.write_text(json.dumps(data));mapping,provenance=load_reasoning_source(p,rows,manifest)
            self.assertEqual(set(mapping),{0})
            self.assertEqual(provenance['excluded_train_indices'],[1])
            self.assertEqual(training_row(rows[2],2,'validation',mapping),rows[2])
            for mutation in ('validation','overlap','unaccounted','not_conflict'):
                bad=copy.deepcopy(data)
                if mutation=='validation':bad['excluded_records'][0]['id']=2
                if mutation=='overlap':bad['excluded_records'][0]['id']=0
                if mutation=='unaccounted':bad['excluded_records']=[]
                if mutation=='not_conflict':bad['excluded_records'][0]['evidence_status']='limited'
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):load_reasoning_source(p,rows,manifest)

    def test_default_server_contract(self):
        c=validate_config(json.loads((Path(__file__).resolve().parents[1]/'configs/local_reasoning.json').read_text()))
        self.assertEqual(c['model_file'],'Qwen3.8-27B-UD-IQ3_XXS.gguf')
        command=server_args(Path('server'),[Path('model'),Path('vision')],c,1234)
        for flag,value in [('--host','127.0.0.1'),('--parallel','1'),('--mmproj','vision'),('--ctx-size','32768'),('--reasoning','on'),('--reasoning-budget','-1')]:
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertNotIn('--cache-reuse',command)

    def test_mock_generation_resume_and_complete_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); images=root/'images'; paths=[]
            for camera in CAMERAS:
                image=images/camera/'a.jpg';image.parent.mkdir(parents=True);image.write_bytes(b'\xff\xd8\xffmock')
                paths.append(str(image))
            rows=[{'problem':'Question?', 'solution':'<answer>Stop.</answer>', 'thinking':'<think>old</think>',
                   'image':paths,'metadata':{'scene_id':str(i),'frame_id':str(i),'category':'behavior'}} for i in range(5)]
            data=root/'data.json';data.write_text(json.dumps(rows))
            split=root/'split.json';manifest=load_split(rows,split,create=True)
            out=root/'out'
            argv=['local_reasoning','--data',str(data),'--images',str(images),'--split',str(split),
                  '--output',str(out),'--skip-tuning','--limit','1']
            @contextmanager
            def fake_server(*args):yield 'http://localhost:1234'
            response={'choices':[{'finish_reason':'stop','message':{'reasoning_content':'Checked the cameras.',
                      'content':json.dumps({'explanation':'The obstacle ahead requires stopping.', 'evidence_status':'grounded'})}}]}
            with patch('local_reasoning.bootstrap',return_value=(Path('server'),[],{},{})), \
                 patch('local_reasoning.serving',fake_server), patch('local_reasoning.web_json',return_value=response) as call:
                with patch.object(sys,'argv',argv):main()
                self.assertEqual(call.call_count,1)
                self.assertTrue(call.call_args.args[1]['chat_template_kwargs']['enable_thinking'])
                with self.assertRaises(ValueError):load_reasoning_source(out/'reasoning_overlay.json',rows,manifest)
                with patch.object(sys,'argv',argv[:-2]):main()
                self.assertEqual(call.call_count,len(manifest['indices']['train']))
                with patch.object(sys,'argv',argv[:-2]):main()
                self.assertEqual(call.call_count,len(manifest['indices']['train']))
            mapping,_=load_reasoning_source(out/'reasoning_overlay.json',rows,manifest)
            self.assertEqual(set(mapping),set(manifest['indices']['train']))

if __name__=='__main__':unittest.main()
