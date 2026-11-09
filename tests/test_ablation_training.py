"""GPU-free regressions for opt-in ablations and interrupted-run recovery."""
import ast
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_config as cfg
from data_utils import write_json


def trainer_method(name, scope):
    tree = ast.parse((ROOT / 'scripts/vlm-finetuning.py').read_text())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenVLTrainer')
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'trainer_method', 'exec'), scope)
    return scope[name]


class AblationTrainingTests(unittest.TestCase):
    def test_c1_defaults_stay_strict_while_ablation_explicitly_allows_changes(self):
        original = cfg.load_experiment()
        fingerprint = cfg.config_hash(original)
        for key in ('train_on_responses_only', 'finetune_attention_modules'):
            modified = json.loads(json.dumps(original))
            modified['common'][key] = True
            with self.assertRaises(ValueError):
                cfg.validate_config(modified)
            modified['experiment_profile'] = 'ablation'
            cfg.validate_config(modified)
            self.assertTrue(cfg.backend_settings(modified, 'cuda')[key])
            with self.assertRaises(ValueError):
                cfg.backend_settings(modified, 'mlx')
            modified['common'][key] = 'false'
            with self.assertRaises(ValueError):
                cfg.validate_config(modified)
        self.assertEqual(cfg.config_hash(original), fingerprint)

    def test_loss_mask_reaches_collator_with_qwen_role_markers(self):
        settings = cfg.backend_settings(cfg.load_experiment(), 'cuda')
        settings.update(max_steps=-1, logging_steps=10, save_total_limit=3, use_wandb=False)
        seen = []
        scope = {
            'logger': logging.getLogger(__name__),
            'FastVisionModel': types.SimpleNamespace(for_training=lambda model: None),
            'UnslothVisionDataCollator': lambda *a, **k: seen.append(k) or k,
            'vision_collator_settings': cfg.vision_collator_settings,
            'SFTConfig': lambda **k: k,
            'SFTTrainer': lambda **k: k,
            'is_bf16_supported': lambda: True,
        }
        setup = trainer_method('setup_trainer', scope)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            'transformers': types.SimpleNamespace(EarlyStoppingCallback=lambda **k: k)
        }):
            trainer = types.SimpleNamespace(config=settings, output_dir=Path(directory), model=object(),
                                            tokenizer=object(), train_dataset=[], val_dataset=[])
            setup(trainer)
            self.assertFalse(seen[-1]['train_on_responses_only'])
            self.assertNotIn('response_part', seen[-1])
            settings['train_on_responses_only'] = True
            setup(trainer)
            self.assertTrue(seen[-1]['train_on_responses_only'])
            self.assertEqual(seen[-1]['instruction_part'], '<|im_start|>user\n')
            self.assertEqual(seen[-1]['response_part'], '<|im_start|>assistant\n')
            self.assertEqual(seen[-1]['max_seq_length'], 2048)
            self.assertEqual(seen[-1]['resize'], (224, 224))

    def make_resume(self, root):
        checkpoint = root / 'checkpoint-12'
        checkpoint.mkdir()
        for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_config.json', 'adapter_model.safetensors'):
            (checkpoint / name).touch()
        write_json(checkpoint / 'trainer_state.json', {'global_step': 12})
        config = dict(experiment_sha256='fixed', mode='reason', max_steps=-1,
                      data_path=str(root / 'data.json'), base_image_path=str(root / 'images'),
                      split_path=str(root / 'input_split.json'))
        write_json(root / 'training_config.json', config)
        for path in (root / 'input_split.json', root / 'split_manifest.json'):
            write_json(path, {'indices': {'train': [0], 'validation': [1], 'test': [2]}})
        return config, checkpoint

    def test_attention_ablation_reaches_adapter_without_enabling_vision(self):
        settings = cfg.backend_settings(cfg.load_experiment(), 'cuda')
        settings['finetune_attention_modules'] = True
        calls = []
        model = types.SimpleNamespace(parameters=lambda: [])
        scope = {
            'logger': logging.getLogger(__name__), 'write_json': write_json,
            'resolve_model_source': lambda *a: ('snapshot', {}),
            'runtime_metadata': lambda *a: {}, 'is_bf16_supported': lambda: True,
            'FastVisionModel': types.SimpleNamespace(
                from_pretrained=lambda *a, **k: (model, object()),
                get_peft_model=lambda *a, **k: calls.append(k) or model),
        }
        load_model = trainer_method('_load_model', scope)
        with tempfile.TemporaryDirectory() as directory:
            owner = types.SimpleNamespace(config=settings, output_dir=Path(directory), _log_gpu_info=lambda: None)
            load_model(owner)
            self.assertTrue(calls[0]['finetune_attention_modules'])
            self.assertFalse(calls[0]['finetune_vision_layers'])
            self.assertTrue(calls[0]['finetune_mlp_modules'])
            self.assertEqual(owner.config['actual_loss_policy'], 'full_text_image_and_padding_masked')

    def test_resume_accepts_only_complete_same_condition_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, checkpoint = self.make_resume(root)
            self.assertEqual(cfg.validate_resume_checkpoint(config, root, checkpoint), str(checkpoint))
            for key, value in [('experiment_sha256', 'changed'), ('mode', 'simple'), ('max_steps', 20),
                               ('data_path', str(root / 'different.json'))]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    cfg.validate_resume_checkpoint(dict(config, **{key: value}), root, checkpoint)
            (checkpoint / 'optimizer.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'optimizer.pt'):
                cfg.validate_resume_checkpoint(config, root, checkpoint)

    def test_resume_rejects_new_split_completed_and_foreign_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, checkpoint = self.make_resume(root)
            with self.assertRaisesRegex(ValueError, 'direct child'):
                cfg.validate_resume_checkpoint(config, root, root.parent / 'other')
            write_json(root / 'completed.json', {'smoke': False})
            with self.assertRaisesRegex(ValueError, 'already complete'):
                cfg.validate_resume_checkpoint(config, root, checkpoint)
            (root / 'completed.json').unlink()
            write_json(root / 'input_split.json', {'indices': {'train': [2], 'validation': [1], 'test': [0]}})
            with self.assertRaisesRegex(ValueError, 'split differs'):
                cfg.validate_resume_checkpoint(config, root, checkpoint)

    def test_training_saves_metrics_state_and_passes_resume(self):
        scope = {'logger': logging.getLogger(__name__), 'write_json': write_json,
                 'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(max_memory_allocated=lambda: 123))}
        train = trainer_method('train', scope)
        calls = []
        metrics = {'train_loss': 0.5, 'train_runtime': 30.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = types.SimpleNamespace(
                train=lambda **k: calls.append(k) or types.SimpleNamespace(metrics=metrics, training_loss=.5),
                save_model=lambda path: calls.append('model'), save_state=lambda: calls.append('state'),
                state=types.SimpleNamespace(global_step=20, epoch=3, best_model_checkpoint='checkpoint-20', best_metric=.4))
            owner = types.SimpleNamespace(trainer=trainer, output_dir=root,
                tokenizer=types.SimpleNamespace(save_pretrained=lambda path: None),
                config=dict(resume_from_checkpoint='checkpoint-12', max_steps=-1, use_wandb=False))
            train(owner)
            self.assertEqual(calls[0], {'resume_from_checkpoint': 'checkpoint-12'})
            self.assertIn('state', calls)
            self.assertEqual(json.loads((root / 'training_metrics.json').read_text()), metrics)
            self.assertEqual(json.loads((root / 'completed.json').read_text())['global_step'], 20)


if __name__ == '__main__':
    unittest.main()
