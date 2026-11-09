"""Deterministic, GPU-free checks for ablation metrics and inference prompts."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import ablation_infer as inference
import ablation_study as study
from ablation_infer import fixed_metrics, public_prompt
from data_utils import generated_answer, write_json, dataset_hash
from experiment_config import load_experiment, config_hash


class AblationInferenceTests(unittest.TestCase):
    def rows(self):
        return [
            {'metadata': {'category': category, 'original_qa_pair': {'tag': tags}}}
            for category, tags in [('behavior', [0]), ('behavior', [0]), ('planning', [0, 1]),
                                   ('perception', [1]), ('behavior', [0])]
        ]

    def response(self, identifier, reference, text, status='ok', finish='eos'):
        return dict(id=str(identifier), gt_answer=reference, answer=text,
                    status=status, finish_reason=finish)

    def test_fixed_denominator_keeps_wrong_invalid_and_failed_responses(self):
        stop = 'The ego vehicle is not moving.'
        responses = [
            self.response(0, stop, '<think>Stop.</think><answer>' + stop + '</answer>'),
            self.response(1, 'Move.', '<answer>move.</answer>'),
            self.response(2, 'Yes.', '<answer>Yes.', status='invalid_format', finish='length'),
            self.response(3, 'Yes.', '<answer>Yes.</answer>'),
            self.response(4, stop, '<answer>' + stop + '</answer>', status='error'),
        ]
        result = fixed_metrics(responses, self.rows())
        self.assertEqual(result['fixed_tag0'], {'correct': 1, 'n': 4, 'accuracy': .25})
        self.assertEqual(result['behavior'], {'correct': 1, 'n': 3, 'accuracy': 1 / 3})
        self.assertEqual(result['stop'], {'correct': 1, 'n': 2, 'accuracy': .5})
        self.assertEqual(result['total'], 5)
        self.assertEqual(result['valid'], 3)
        self.assertEqual(result['length'], 1)

    def test_different_predictions_cannot_change_fixed_denominator(self):
        correct = [self.response(i, 'Yes.', '<answer>Yes.</answer>') for i in range(5)]
        failed = [self.response(i, 'Yes.', '', status='error') for i in range(5)]
        a, b = fixed_metrics(correct, self.rows()), fixed_metrics(failed, self.rows())
        for group in ('fixed_tag0', 'behavior', 'stop'):
            self.assertEqual(a[group]['n'], b[group]['n'])
        self.assertEqual(a['fixed_tag0']['correct'], 4)
        self.assertEqual(b['fixed_tag0']['correct'], 0)

    def test_scoring_uses_strict_generated_text_not_cached_final_answer(self):
        response = self.response(0, 'Stop.', '<answer>Go.</answer>')
        response['final_answer'] = 'Stop.'
        self.assertEqual(fixed_metrics([response], self.rows())['fixed_tag0']['correct'], 0)
        response['answer'] = 'Here is my answer: <answer>Stop.</answer>'
        self.assertEqual(fixed_metrics([response], self.rows())['fixed_tag0']['correct'], 0)

    def test_absent_categories_use_none_not_zero(self):
        result = fixed_metrics([self.response(3, 'Yes.', '<answer>Yes.</answer>')], self.rows())
        for group in ('fixed_tag0', 'behavior', 'stop'):
            self.assertEqual(result[group], {'correct': 0, 'n': 0, 'accuracy': None})

    def test_reason_and_simple_public_prompts_preserve_question(self):
        question = 'What should the ego vehicle do at this intersection?'
        reason = public_prompt(question, 'reason')
        simple = public_prompt(question, 'simple')
        self.assertIn(question, reason)
        self.assertIn(question, simple)
        self.assertIn('Put your reasoning within <think></think> tags', reason)
        self.assertIn('Put only your final answer within <answer></answer> tags.', simple)
        self.assertNotIn('<think>', simple)
        self.assertNotIn('provide reasoning', simple)

    def test_strict_parser_accepts_reason_answer_and_rejects_malformed_outputs(self):
        self.assertEqual(generated_answer('<think>Traffic is stopped.</think>\n<answer>Stop.</answer>'), 'Stop.')
        self.assertEqual(generated_answer('<answer>Stop.</answer>'), 'Stop.')
        for invalid in ['Stop.', '<think>Stop.</think>', '<answer>Stop.',
                        '<answer>Stop.</answer><answer>Go.</answer>',
                        '<think><answer>Stop.</answer></think>',
                        '<answer>Stop.</answer> outside text', '<answer></answer>']:
            with self.subTest(text=invalid), self.assertRaises(ValueError):
                generated_answer(invalid)

    def test_orphaned_results_metrics_context_and_proof_are_never_adopted(self):
        for suffix in ('', '.metrics.json', '.context.json', '.complete.json'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'responses.json'
                Path(str(output) + suffix).write_text('[]')
                with self.assertRaisesRegex(ValueError, 'original metadata'):
                    inference.ensure_metadata(output, {'schema_version': 1})
                self.assertFalse(Path(str(output) + '.meta.json').exists())

    def test_changed_adapter_config_or_context_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'responses.json'
            expected = dict(schema_version=1, adapter_config_sha256='adapter', graph_context_sha256='context')
            self.assertFalse(inference.ensure_metadata(output, expected))
            for key in ('adapter_config_sha256', 'graph_context_sha256'):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'provenance changed'):
                    inference.ensure_metadata(output, dict(expected, **{key: 'changed'}))

    def test_legacy_metadata_is_preserved_and_forces_rescore(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'responses.json'
            legacy = dict(schema_version=1, adapter_sha256='weights', split='validation')
            write_json(Path(str(output) + '.meta.json'), legacy)
            write_json(output, [])
            expected = dict(legacy, adapter_config_sha256='adapter', graph_context_sha256=None)
            self.assertTrue(inference.ensure_metadata(output, expected))
            self.assertEqual(json.loads(Path(str(output) + '.meta.legacy.json').read_text()), legacy)
            self.assertTrue(Path(str(output) + '.provenance_upgrade.json').is_file())
            self.assertFalse(inference.verified_complete(output, False))

    def test_resume_checks_ids_question_truth_and_complete_row_count(self):
        row = dict(problem='Action?', solution='<answer>Stop.</answer>')
        expected = [(3, row, True)]
        response = dict(id='3', question='Action?', gt_answer='Stop.', answer='<answer>Stop.</answer>',
                        final_answer='Stop.', status='ok', scored=True)
        inference.validate_response_prefix([response], expected, complete=True)
        inference.validate_response_prefix([], expected)
        with self.assertRaisesRegex(ValueError, 'incomplete response set'):
            inference.validate_response_prefix([], expected, complete=True)
        for key, value in [('id', '0'), ('question', 'Changed?'), ('gt_answer', 'Go.'), ('scored', False),
                           ('final_answer', 'Go.')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                inference.validate_response_prefix([dict(response, **{key: value})], expected)
        with self.assertRaisesRegex(ValueError, 'prefix length'):
            inference.validate_response_prefix([response, response], expected)

    def test_completion_proof_detects_response_and_metric_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'responses.json'
            for path in (output, Path(str(output) + '.meta.json'), Path(str(output) + '.metrics.json')):
                write_json(path, {})
            proof = Path(str(output) + '.complete.json')
            write_json(proof, inference.completion_proof(output, False))
            self.assertTrue(inference.verified_complete(output, False))
            write_json(output, {'changed': True})
            with self.assertRaisesRegex(ValueError, 'artifacts changed'):
                inference.verified_complete(output, False)
            write_json(output, {})
            Path(str(output) + '.metrics.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing a recorded artifact'):
                inference.verified_complete(output, False)

    def test_legacy_completed_validation_rescores_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root / 'model'; model.mkdir()
            output = root / 'responses.json'; data = root / 'data.json'
            rows = [dict(problem='Action?', solution='<answer>Stop.</answer>',
                         metadata=dict(category='behavior', original_qa_pair={'tag': [0]}))]
            config = dict(mode='reason', model_source={'model_id': 'test/model', 'resolved_revision': 'a' * 40})
            split = dict(seed=3407, val_size=.1, test_size=.1, indices={'validation': [0], 'test': []})
            write_json(data, rows); write_json(model/'training_config.json', config)
            write_json(model/'split_manifest.json', split); write_json(model/'completed.json', {'smoke': False})
            write_json(model/'adapter_config.json', {}); (model/'adapter_model.safetensors').write_bytes(b'weights')
            response = dict(id='0', question='Action?', gt_answer='Stop.', answer='<answer>Stop.</answer>',
                            final_answer='Stop.', status='ok', scored=True, finish_reason='eos')
            write_json(output, [response]); write_json(Path(str(output)+'.metrics.json'), {'obsolete': True})
            legacy = dict(schema_version=1, split='validation', indices=[0], dataset_sha256=dataset_hash(rows),
                          adapter_sha256=inference.sha(model/'adapter_model.safetensors'), mode='reason',
                          prompt='current', do_sample=False, seed=3407, budget=512, model_source=config['model_source'],
                          training_config_sha256=inference.sha(model/'training_config.json'),
                          selection_metric='fixed_tag0 exact accuracy; no graph gating, all failures retained')
            write_json(Path(str(output)+'.meta.json'), legacy)
            argv = ['inference', '--model_dir', str(model), '--data_path', str(data), '--image_root', str(root),
                    '--output', str(output), '--split', 'validation']
            with patch.object(sys, 'argv', argv), patch.object(inference, 'load_split'), \
                    patch.object(inference, 'make_predictor', side_effect=AssertionError('GPU must not load')), \
                    contextlib.redirect_stdout(io.StringIO()):
                inference.main()
                metrics = json.loads(Path(str(output)+'.metrics.json').read_text())
                self.assertEqual(metrics['fixed_tag0']['correct'], 1)
                self.assertTrue(metrics['recomputed_without_generation'])
                with patch.object(inference, 'fixed_metrics', side_effect=AssertionError('Proven completion must skip')):
                    inference.main()
            self.assertTrue(inference.verified_complete(output, False))


class AblationRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        model = root/'models'/'reason'; model.mkdir(parents=True)
        config = load_experiment()
        path = root/'config.json'; write_json(path, config)
        variant = dict(config=str(path), mode='reason')
        write_json(model/'training_config.json', dict(experiment_sha256=config_hash(config), mode='reason'))
        (model/'partial.txt').write_text('retain this evidence')
        return model, variant

    def test_first_epoch_failure_is_archived_and_same_config_can_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model, variant=self.fixture(root)
            archived=study.archive_uncheckpointed_attempt(root, model, variant)
            self.assertFalse(model.exists())
            self.assertEqual((archived/'partial.txt').read_text(), 'retain this evidence')
            self.assertTrue((archived/'restart_record.json').exists())
            self.assertEqual(archived.parent, root/'failed_attempts')
            self.assertIsNone(study.archive_uncheckpointed_attempt(root, model, variant))

    def test_completed_or_checkpoint_attempts_are_never_archived(self):
        for name in ('completed.json', 'checkpoint-1'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); model, variant=self.fixture(root)
                (model/name).touch()
                self.assertIsNone(study.archive_uncheckpointed_attempt(root, model, variant))
                self.assertTrue(model.exists())
                self.assertFalse((root/'failed_attempts').exists())

    def test_changed_or_unknown_attempt_is_not_automatically_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model, variant=self.fixture(root)
            write_json(model/'training_config.json', dict(experiment_sha256='changed', mode='reason'))
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                study.archive_uncheckpointed_attempt(root, model, variant)
            (model/'training_config.json').unlink()
            with self.assertRaisesRegex(ValueError, 'no training provenance'):
                study.archive_uncheckpointed_attempt(root, model, variant)
            self.assertTrue(model.exists())

    def test_report_only_does_not_prepare_execute_or_change_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'plan.json'
            write_json(path, {'variants': [], 'prompt_diagnostics': []})
            before=path.read_bytes()
            with patch.object(sys, 'argv', ['study','--run_dir', str(root), '--report_only']), \
                    patch.object(study, 'prepare', side_effect=AssertionError('Must not prepare')), \
                    patch.object(study, 'execute', side_effect=AssertionError('Must not execute')), \
                    contextlib.redirect_stdout(io.StringIO()):
                study.main()
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue((root/'REPORT.md').exists())
            self.assertFalse((root/'selection.json').exists())


if __name__ == '__main__':
    unittest.main()
