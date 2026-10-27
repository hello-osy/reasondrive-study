# ============================================
# 실험용 확장·설정 변경: GPU/가중치 없이 실행 조건·Instruct 경로 검증
# ============================================
import argparse
import ast
import contextlib
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_config as cfg
import eval_finetuned_models as inference


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        row, col = key
        if isinstance(row, int):
            return types.SimpleNamespace(item=lambda: self.rows[row][col])
        return Tensor([values[col] for values in self.rows[row]])


class Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=Tensor([[1, 2]]))
        self.input_ids = self['input_ids']

    def to(self, _device):
        return self


class TrainingContractTests(unittest.TestCase):
    def test_direct_trainer_defaults_and_runner_share_3b_config(self):
        tree = ast.parse((ROOT / 'scripts/vlm-finetuning.py').read_text())
        defaults = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CONFIG' for t in node.targets))
        method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
        scope = {'argparse': argparse, 'DEFAULT_CONFIG': ast.literal_eval(defaults),
                 **{name: getattr(cfg, name) for name in ('DEFAULT_PATH', 'load_experiment', 'backend_settings', 'override_experiment')}}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'parse_args', 'exec'), scope)
        with patch.object(sys, 'argv', ['trainer']):
            args = scope['parse_args']()
        expected = cfg.backend_settings(cfg.load_experiment(), 'cuda')
        self.assertEqual(args.model_id, expected['model_id'])
        self.assertEqual(args.model_revision, expected['model_revision'])
        self.assertEqual(args.batch_size, 1)
        self.assertFalse(args.train_on_responses_only)

    def test_unpinned_models_and_loss_policy_drift_rejected(self):
        config = cfg.load_experiment()
        with self.assertRaises(ValueError):
            cfg.override_experiment(config, 'cuda', {'model_id': 'other/model'})
        with self.assertRaises(ValueError):
            cfg.override_experiment(config, 'cuda', {'model_revision': 'main'})
        with self.assertRaises(ValueError):
            cfg.override_experiment(config, 'mlx', {'train_on_responses_only': True})
        changed = cfg.override_experiment(config, 'cuda', {'epochs': 1})
        self.assertNotEqual(cfg.config_hash(changed), cfg.config_hash(config))
        self.assertEqual(config['common']['epochs'], 3)

    def test_snapshot_revision_must_be_verified(self):
        revision = cfg.backend_settings(cfg.load_experiment(), 'cuda')['model_revision']
        with tempfile.TemporaryDirectory() as directory:
            pinned = Path(directory) / 'snapshots' / revision
            fake = types.SimpleNamespace(snapshot_download=lambda *a, **k: str(pinned))
            with patch.dict(sys.modules, {'huggingface_hub': fake}):
                path, source = cfg.resolve_model_source('test/model', revision)
                self.assertEqual(source['resolved_revision'], revision)
                self.assertEqual(source['processor_revision'], revision)
                fake.snapshot_download = lambda *a, **k: str(Path(directory) / 'unknown')
                with self.assertRaises(ValueError):
                    cfg.resolve_model_source('test/model', revision)

    def test_smoke_unknown_identity_and_changed_conditions_rejected(self):
        experiment = cfg.load_experiment()
        settings = cfg.backend_settings(experiment, 'cuda')
        source = {'model_id': settings['model_id'], 'resolved_revision': settings['model_revision'],
                  'processor_revision': settings['model_revision']}
        config = {'max_steps': -1, 'experiment_sha256': cfg.config_hash(experiment), 'model_source': source}
        manifest = {'indices': {'test': [1]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'split_manifest.json').write_text(json.dumps(manifest))
            (root / 'completed.json').write_text(json.dumps({'smoke': False}))
            inference.validate_training_run(config, manifest, root, experiment)
            for changed in [dict(config, max_steps=1), dict(config, model_source={}),
                            dict(config, experiment_sha256='different')]:
                with self.assertRaises(ValueError):
                    inference.validate_training_run(changed, manifest, root, experiment)

    def test_instruct_uses_no_adapter_and_preserves_test_context_budget(self):
        experiment = cfg.load_experiment()
        settings = cfg.backend_settings(experiment, 'cuda')
        calls, prompts = [], []
        model = types.SimpleNamespace(generate=lambda **kw: (calls.append(kw) or Tensor([[1, 2, 3, 9]])))
        class Processor:
            tokenizer = types.SimpleNamespace(eos_token_id=9)
            def apply_chat_template(self, messages, **kwargs):
                prompts.append(messages)
                return 'prompt'
            def __call__(self, **kwargs):
                return Inputs()
            def batch_decode(self, *args, **kwargs):
                return ['<answer>Stop</answer>']
        processor = Processor()
        modules = {
            'unsloth': types.SimpleNamespace(FastVisionModel=types.SimpleNamespace(
                from_pretrained=lambda *a, **k: (model, processor), for_inference=lambda model: None)),
            'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True),
                backends=types.SimpleNamespace(cudnn=types.SimpleNamespace()), inference_mode=contextlib.nullcontext),
            'transformers': types.SimpleNamespace(set_seed=lambda seed: None),
            'qwen_vl_utils': types.SimpleNamespace(process_vision_info=lambda messages: (['image'], None)),
            'peft': types.SimpleNamespace(PeftModel=types.SimpleNamespace(
                from_pretrained=lambda *a, **k: self.fail('Instruct must not load an adapter')))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / 'camera.jpg'
            photo.touch()
            row = {'image': [str(photo)], 'problem': 'Action?', 'solution': '<answer>Stop</answer>'}
            context = {'sha256': 'context_hash', 'items': [dict(row, id='graph:s:f')]}
            manifest = {'dataset_sha256': 'dataset', 'indices': {'test': [0]}}
            path = root / 'instruct.json'
            with patch.dict(sys.modules, modules), patch.object(inference, 'runtime_metadata', return_value={}), \
                    patch.object(inference, 'resolve_model_source', return_value=('snapshot', {'resolved_revision': settings['model_revision']})):
                results = inference.evaluate_model(None, [row], manifest, root, str(path), 'instruct', experiment, context, root / 'graph_context.json')
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['status'], 'ok')
            self.assertEqual(results[0]['finish_reason'], 'eos')
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call['max_new_tokens'] == settings['max_new_tokens'] for call in calls))
            self.assertTrue(all(call['do_sample'] is False for call in calls))
            self.assertTrue(all(len(message) == 1 for message in prompts))
            self.assertIn('Provide only your final answer', prompts[0][0]['content'][-1]['text'])
            meta = json.loads(Path(str(path) + '.meta.json').read_text())
            self.assertEqual((meta['mode'], meta['prompt_mode']), ('instruct', 'simple'))
            sidecar = json.loads(Path(str(path) + '.context.json').read_text())
            self.assertFalse(sidecar['responses'][0]['scored'])
            self.assertEqual(sidecar['graph_context']['sha256'], 'context_hash')


if __name__ == '__main__':
    unittest.main()
