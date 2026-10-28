import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import continue_v6_experiment as continuation
from data_utils import write_json
from evaluate import _experiment_conditions
from experiment_config import load_experiment, config_hash


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'old'
        self.config = load_experiment()
        self.config['common'].update(batch_size=40, gradient_accumulation_steps=1)
        model = self.config['backends']['cuda']
        self.model_source = {'model_id': model['model_id'], 'resolved_revision': model['model_revision'],
                             'processor_revision': model['model_revision']}
        self.images = Path(self.tmp.name) / 'images'
        self.images.mkdir()
        self.data = Path(self.tmp.name) / 'data.json'
        write_json(self.data, [])
        write_json(self.source / 'status.json', {'profile': 'modified', 'status': 'failed'})
        for name in ('split_manifest.json', 'simple/split_manifest.json', 'simple/graph_context.json', 'input_repair_report.json'):
            write_json(self.source / name, {})
        write_json(self.source / 'simple/completed.json', {'smoke': False})
        write_json(self.source / 'simple/training_config.json', {
            'mode': 'simple', 'experiment': self.config, 'experiment_sha256': config_hash(self.config),
            'model_source': self.model_source, 'data_path': str(self.data), 'base_image_path': str(self.images)})
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            (self.source / 'simple' / name).write_text('fixture')

    def test_reuses_simple_without_mutating_original_configuration(self):
        _, original, changed, _, _ = continuation.plan(self.source)
        self.assertEqual(original['common']['batch_size'], 40)
        self.assertEqual(changed['common']['batch_size'], 30)
        self.assertEqual(json.loads((self.source / 'simple/training_config.json').read_text())['experiment'], original)

    def test_refuses_to_discard_saved_reason_progress(self):
        (self.source / 'reason/checkpoint-106').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'saved training progress'):
            continuation.plan(self.source)

    def test_only_reason_training_and_two_model_evaluation_are_dispatched(self):
        with patch.object(sys, 'argv', ['continue', str(self.source)]), \
             patch.object(continuation, 'start'), patch.object(continuation, 'run_step') as step, \
             patch.object(continuation, 'complete'), patch.object(continuation, 'summarize'):
            continuation.main()
        self.assertEqual([call.args[1] for call in step.call_args_list],
                         ['train_reason', 'evaluate_simple', 'evaluate_reason', 'score'])
        run = step.call_args_list[0].args[0]
        self.assertFalse((run / 'simple').is_symlink())
        self.assertEqual((run / 'simple/adapter_model.safetensors').read_bytes(),
                         (self.source / 'simple/adapter_model.safetensors').read_bytes())
        self.assertIn('--allow_batching_difference', step.call_args_list[-1].args[2])
        self.assertIn(run / 'simple_experiment.json', step.call_args_list[1].args[2])

    def test_batch_exception_keeps_other_conditions_strict_and_validates_hash(self):
        meta = {'backend': 'cuda', 'image_size': 224, 'model_source': self.model_source,
                'experiment': self.config, 'experiment_sha256': config_hash(self.config)}
        changed = copy.deepcopy(meta)
        changed['experiment']['common']['batch_size'] = 30
        changed['experiment_sha256'] = config_hash(changed['experiment'])
        self.assertNotEqual(_experiment_conditions(meta), _experiment_conditions(changed))
        self.assertEqual(_experiment_conditions(meta, True), _experiment_conditions(changed, True))
        changed['experiment']['common']['learning_rate'] = .0001
        with self.assertRaises(ValueError):
            _experiment_conditions(changed, True)
        changed['experiment_sha256'] = config_hash(changed['experiment'])
        self.assertNotEqual(_experiment_conditions(meta, True), _experiment_conditions(changed, True))
        with self.assertRaises(ValueError):
            _experiment_conditions({}, True)


if __name__ == '__main__':
    unittest.main()
