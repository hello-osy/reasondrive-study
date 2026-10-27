# ============================================
# 실험용 확장·설정 변경 — 파일 전체
# 결과 정렬·실패 보존·완료 산출물 검증. 학습과 API는 호출하지 않는다.
# ============================================
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_experiment as report
from data_utils import write_json
from experiment_config import load_experiment


class ReportExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def row(self, scene='scene_with_underscores', frame='frame_a', question='Question?'):
        return {'problem': question, 'solution': '<answer>Stop</answer>',
                'metadata': {'scene_id': scene, 'frame_id': frame}}

    def raw(self, row, position, answer='<answer>Stop</answer>'):
        return {'id': f"{row['metadata']['scene_id']}_{row['metadata']['frame_id']}_{position}",
                'question': '<image>\n' + row['problem'], 'gt_answer': 'Stop', 'answer': answer}

    def test_original_ids_map_by_position_not_question_or_underscores(self):
        row = self.row()
        records = [('7', row, True), ('8', row, True), ('graph:s:f', row, False)]
        raw = [self.raw(row, 2), self.raw(row, 0)]
        result = report.normalize_public(raw, records)
        self.assertEqual([r['id'] for r in result], ['7', '8', 'graph:s:f'])
        self.assertEqual([r['status'] for r in result], ['ok', 'error', 'ok'])
        self.assertFalse(result[2]['scored'])
        self.assertEqual(result[1]['answer'], '')

    def test_malformed_answers_kept_raw_and_invalid(self):
        row = self.row();raw = self.raw(row, 0, '<think><answer>Stop</answer></think>')
        actual = report.normalize_public([raw], [('1', row, True)])[0]
        self.assertEqual(actual['status'], 'invalid_format')
        self.assertEqual(actual['answer'], raw['answer'])
        self.assertEqual(actual['final_answer'], '')

    def test_misaligned_duplicate_and_unknown_rows_rejected(self):
        row = self.row();raw = self.raw(row, 0)
        for entries in ([raw, raw], [dict(raw, id='unknown')], [dict(raw, gt_answer='Go')],
                        [dict(raw, question='Wrong')]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                report.normalize_public(entries, [('1', row, True)])

    def test_public_settings_preserve_original_scientific_conditions(self):
        config = load_experiment()
        effective = report.public_settings(config)
        self.assertEqual(effective['collator_max_seq_length'], 4096)
        self.assertEqual(effective['max_seq_length'], 2048)
        self.assertEqual((effective['train_seed'], effective['eval_split_seed']), (3407, 3047))
        self.assertTrue(effective['do_sample'])
        self.assertIsNone(effective['generation_seed'])
        config['common']['learning_rate'] = 1e-4
        with self.assertRaises(ValueError):report.public_settings(config)

    def completed(self, profile):
        run = self.root / profile / 'test_run'
        protocol = 'public_baseline' if profile == 'public_baseline' else 'c1_scene'
        count = {'total': 2, 'valid_answers': 1, 'status_error': 1, 'invalid_format': 0,
                 'accuracy': 2, 'language': 0, 'match': 0, 'graph_excluded': 0}
        report_json = {'protocol': protocol, 'rows': [
            {'model': mode, 'Accuracy': .5, 'Final_noGPT': None} for mode in ('simple', 'reason')],
            'counts': {mode: count for mode in ('simple', 'reason')}}
        write_json(run / 'comparison/comparison.json', report_json)
        (run / 'comparison/comparison.md').write_text('A synthetic test fixture\n')
        (run / 'comparison/comparison.csv').write_text('model,Accuracy\nsimple,0.5\nreason,0.5\n')
        for file in ('experiment.json', 'source_manifest.json'):
            write_json(run / file, {})
        for mode in ('simple', 'reason'):
            for suffix in ('.json', '.json.meta.json', '.json.context.json'):
                write_json(run / (mode + suffix), {})
            if profile == 'public_baseline':
                for suffix in ('.raw.json', '.json.protocol.json'):write_json(run / (mode + suffix), {})
                write_json(run / mode / 'training_result.json', {'status': 'completed'})
            else:
                write_json(run / mode / 'completed.json', {'smoke': False})
                write_json(run / mode / 'training_config.json', {})
        with contextlib.redirect_stdout(io.StringIO()):report.complete(run, profile)
        return run

    def test_resume_skips_completed_simple_and_dispatches_reason_checkpoint(self):
        run = self.root / 'resume'
        config = load_experiment()
        config['common'].update(batch_size=24, gradient_accumulation_steps=1)
        write_json(run / 'experiment.json', config)
        effective = report.public_settings(config)
        write_json(run / 'public_settings.json', effective)
        data_path = self.root / 'data.json'
        write_json(data_path, [])
        snapshot = str(self.root / 'base')
        checkpoint = run / 'reason' / 'adapter' / 'checkpoint-500'
        for mode in ('simple', 'reason'):
            adapter = run / mode / 'adapter'
            adapter.mkdir(parents=True)
            for name in ('adapter_model.safetensors', 'adapter_config.json'):
                (adapter / name).write_text('fixture')
            write_json(run / mode / 'training_result.json', {
                'status': 'completed' if mode == 'simple' else 'running', 'mode': mode,
                'data_sha256': report.digest(data_path), 'model_path': snapshot,
                'image_root': str(self.root), 'actual_output_dir': str(adapter),
                'effective_training_config': {k: effective[k] for k in (
                    'epochs', 'batch_size', 'gradient_accumulation_steps', 'eval_steps', 'save_steps')}})
        args = types.SimpleNamespace(run_dir=run, config=run/'experiment.json',
                                     data_path=data_path, image_root=self.root, resume=True, resume_batch_size=8)
        with patch('evaluate.public_baseline_manifest', return_value={'test_indices': []}), \
             patch('graph_context.prepare_graph_context', return_value={}), \
             patch('experiment_config.resolve_model_source', return_value=(snapshot, {})), \
             patch('public_baseline.latest_checkpoint', return_value=checkpoint), \
             patch.object(report, 'start'), \
             patch.object(report, 'run_step', side_effect=RuntimeError('dispatch captured')) as step, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'dispatch captured'):
                report.run_public(args)
        step.assert_called_once()
        self.assertEqual(step.call_args.args[1], 'train_reason')
        command = step.call_args.args[2]
        self.assertEqual(command[command.index('--resume_from_checkpoint') + 1], checkpoint)
        self.assertEqual(command[command.index('--batch_size') + 1], 8)
        self.assertEqual(command[command.index('--gradient_accumulation_steps') + 1], 3)
        self.assertIn('--allow_resume_batch_change', command)

    def test_two_tables_are_unpaired_and_absent_metrics_stay_na(self):
        self.completed('public_baseline');self.completed('modified')
        with contextlib.redirect_stdout(io.StringIO()):report.summarize(self.root)
        payload = json.loads((self.root / 'comparison.json').read_text())
        self.assertEqual(len(payload['experiments']), 2)
        self.assertEqual(len(payload['rows']), 4)
        self.assertEqual(payload['comparison_type'], 'descriptive_unpaired')
        self.assertNotIn('delta', payload)
        self.assertIn('N/A', (self.root / 'comparison.csv').read_text())
        self.assertEqual((self.root / 'comparison.md').read_text().count('## 실험'), 2)

    def test_tampering_with_completed_predictions_rejects_reuse(self):
        run = self.completed('modified')
        write_json(run / 'simple.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            report.load_completed(self.root, 'modified')

    def test_missing_profile_does_not_borrow_another_profile_results(self):
        self.completed('modified')
        with contextlib.redirect_stdout(io.StringIO()):report.summarize(self.root)
        payload = json.loads((self.root / 'comparison.json').read_text())
        self.assertEqual(len(payload['experiments']), 1)
        self.assertIn('아직 완료된 실행이 없습니다.', (self.root / 'comparison.md').read_text())

    def test_public_workflow_connects_two_fresh_trainings_and_common_scoring(self):
        from graph_context import ROOT_QUESTION
        from evaluate import compare, LANGUAGE_KEYS
        run = self.root / 'public_baseline' / 'workflow'
        config = load_experiment()
        config['common'].update(batch_size=24, gradient_accumulation_steps=1)
        write_json(run / 'experiment.json', config)
        write_json(run / 'source_manifest.json', {})
        report.state(run, source_sha256='source-fixture', profile='public_baseline')
        data = []
        for i in range(40):
            data.append({'problem': ROOT_QUESTION, 'solution': '<answer>Stop</answer>',
                         'image': ['unused.jpg'], 'metadata': {'scene_id': f's{i}', 'frame_id': f'f{i}',
                         'category': 'perception', 'original_qa_pair': {'Q': ROOT_QUESTION, 'A': 'Stop', 'tag': [2]}}})
        write_json(self.root / 'data.json', data)
        calls = []
        def fake_step(run_dir, stage, arguments):
            calls.append((stage, arguments))
            args = list(map(str, arguments))
            if stage.startswith('train_'):
                out = Path(args[args.index('--output_dir')+1])
                write_json(out / 'training_result.json', {'status': 'completed', 'actual_output_dir': str(out/'adapter'),
                                                         'effective_training_config': config['common']})
            else:
                ordered = json.loads(Path(args[args.index('--data_path')+1]).read_text())
                # Preserve a skipped row in the final denominator.
                raw = [self.raw(row, i) for i, row in enumerate(ordered) if i != 1]
                write_json(args[args.index('--output_path')+1], raw)
        source = {'model_id': config['backends']['cuda']['model_id'],
                  'resolved_revision': config['backends']['cuda']['model_revision'],
                  'processor_revision': config['backends']['cuda']['model_revision']}
        def fake_compare(paths, data, protocol):
            return compare(paths, data, lambda p,r:{k:0.0 for k in LANGUAGE_KEYS}, protocol=protocol)
        args = types.SimpleNamespace(run_dir=run, config=run/'experiment.json',
                                     data_path=self.root/'data.json', image_root=self.root)
        with patch.object(report, 'run_step', side_effect=fake_step), \
             patch('experiment_config.resolve_model_source', return_value=(str(self.root/'base'), source)), \
             patch('experiment_config.runtime_metadata', return_value={}), \
             patch('evaluate.compare', side_effect=fake_compare), contextlib.redirect_stdout(io.StringIO()):
            report.run_public(args)
        self.assertEqual([stage for stage,_ in calls], ['train_simple','train_reason','infer_simple','infer_reason'])
        for _, arguments in calls[:2]:
            self.assertEqual(arguments[arguments.index('--batch_size') + 1], 24)
            self.assertEqual(arguments[arguments.index('--gradient_accumulation_steps') + 1], 1)
            self.assertEqual(arguments[arguments.index('--eval_steps') + 1], 100)
            self.assertEqual(arguments[arguments.index('--save_steps') + 1], 100)
        settings = json.loads((run / 'public_settings.json').read_text())
        self.assertEqual(settings['effective_batch_size'], 24)
        self.assertTrue(settings['batching_override'])
        bases = [str(args[args.index('--model_path')+1]) for _,args in calls[:2]]
        self.assertEqual(bases, [str(self.root/'base')]*2)
        result = report.load_completed(self.root, 'public_baseline')[1]
        self.assertEqual(result['counts']['simple']['total'], 4)
        self.assertEqual(result['counts']['simple']['status_error'], 1)
        self.assertEqual(result['counts']['simple']['language'], 4)


if __name__ == '__main__':unittest.main()
