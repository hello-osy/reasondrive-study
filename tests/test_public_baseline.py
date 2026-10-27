# ============================================
# 실험용 확장·설정 변경 — 원본 실행 연결부의 CPU/mock 검증
# 모델·학습·API를 실행하지 않고 실제 보존 소스의 분할·형식·생성 조건을 검사한다.
# ============================================
import argparse
import ast
import contextlib
import copy
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import typing
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import public_baseline as baseline


class PublicBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.camera1 = self.directory / 'camera1.jpg'
        self.camera2 = self.directory / 'camera2.jpg'
        self.camera1.touch()
        self.camera2.touch()
        self.row = {'image': [str(self.camera2), str(self.camera1)], 'problem': 'Where is the car?',
                    'thinking': '  <thinkThe RAW explanation without normalization.  ',
                    'solution': '  <answer>PRIVATE_REFERENCE</answer>  ',
                    'metadata': {'scene_id': 'scene', 'frame_id': 'frame'}}
        self.data = self.directory / 'data.json'
        self.data.write_text(json.dumps([self.row]))

    def trainer_module(self):
        """Compile only actual upstream methods; importing GPU packages is unnecessary."""
        source = (baseline.VENDOR / 'vlm-finetuning.py').read_text()
        tree = ast.parse(source)
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QwenVLTrainer')
        wanted = ('prepare_data', '_fix_image_paths', '_format_sample', 'setup_trainer')
        original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        defaults = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CONFIG' for t in n.targets))
        module = types.ModuleType('mock_upstream_training')
        dataset_calls = []
        row = self.row
        class Dataset:
            @classmethod
            def from_json(cls, path):
                return cls()
            def __len__(self):
                return 2
            def train_test_split(self, **kwargs):
                dataset_calls.append(kwargs)
                return {'train': [row, dict(row, image=[])], 'test': [row]}
        def config(*, eval_strategy=None, max_length=None, **kwargs):
            return dict(kwargs, eval_strategy=eval_strategy, max_length=max_length)
        def trainer(*, processing_class=None, **kwargs):
            return types.SimpleNamespace(processing_class=processing_class, **kwargs)
        def collator(model, tokenizer, max_seq_length=None, train_on_responses_only=False):
            return {'max_seq_length': max_seq_length, 'train_on_responses_only': train_on_responses_only}
        module.__dict__.update({name: getattr(typing, name) for name in ('Any', 'Dict', 'List', 'Union')})
        module.__dict__.update(os=os, Path=Path, Dataset=Dataset, tqdm=lambda rows: rows,
            logger=logging.getLogger('public-baseline-test'), SFTConfig=config, SFTTrainer=trainer,
            UnslothVisionDataCollator=collator,
            FastVisionModel=types.SimpleNamespace(for_training=lambda model: None),
            is_bf16_supported=lambda: True,
            torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
        exec(compile(ast.Module(body=[original], type_ignores=[]), '<upstream-methods>', 'exec'), module.__dict__)
        module.DEFAULT_CONFIG = ast.literal_eval(defaults)
        module.dataset_calls = dataset_calls
        def init(self, config):
            self.config = config
            self.model = object()
            self.tokenizer = object()
            self.output_dir = Path(config['output_dir']) / 'original_timestamp_directory'
        module.QwenVLTrainer.__init__ = init
        module.QwenVLTrainer.train = lambda self: None
        module.QwenVLTrainer.run_inference_example = lambda self: None
        return module

    def inference_module(self, mode):
        source = (baseline.VENDOR / 'eval_finetuned_models.py').read_text()
        if mode == 'simple':
            source = baseline.derive_simple_inference(source)
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_model')
        prompts, calls = [], []
        class Output:
            def __getitem__(self, key):
                return [[7]]
        class Inputs(dict):
            input_ids = types.SimpleNamespace(shape=(1, 2))
            def to(self, device):
                return self
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                prompts.append(copy.deepcopy(messages))
                return 'serialized user prompt'
            def __call__(self, *args, **kwargs):
                return Inputs()
            def batch_decode(self, *args, **kwargs):
                return ['<answer>raw prediction</answer>']
        def generate(**kwargs):
            calls.append(kwargs)
            return Output()
        model = types.SimpleNamespace(generate=generate)
        module = types.ModuleType('mock_upstream_inference')
        module.__dict__.update(os=os, json=json, tqdm=lambda rows, **kwargs: rows,
            torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), no_grad=contextlib.nullcontext),
            FastVisionModel=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: (model, Processor()),
                                                for_inference=lambda model: None),
            process_vision_info=lambda messages: (['images'], None))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<upstream-inference>', 'exec'), module.__dict__)
        return module, prompts, calls

    def test_manifest_detects_source_and_commit_changes(self):
        baseline.verify_vendor()
        vendor = self.directory / 'vendor'
        shutil.copytree(baseline.VENDOR, vendor)
        (vendor / 'vlm-finetuning.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            baseline.verify_vendor(vendor)
        manifest = json.loads((vendor / 'manifest.json').read_text())
        manifest['upstream_commit'] = 'other'
        (vendor / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'upstream commit'):
            baseline.verify_vendor(vendor)

    def test_keyword_aliases_change_spelling_only_when_signature_requires_it(self):
        patches = []
        def new_api(*, eval_strategy, max_length, untouched):
            return eval_strategy, max_length, untouched
        adapted = baseline.adapt_keywords(new_api,
            {'evaluation_strategy': 'eval_strategy', 'max_seq_length': 'max_length'}, patches, 'SFTConfig')
        self.assertEqual(adapted(evaluation_strategy='steps', max_seq_length=2048, untouched=10), ('steps', 2048, 10))
        self.assertEqual([(p['from'], p['to']) for p in patches],
                         [('evaluation_strategy', 'eval_strategy'), ('max_seq_length', 'max_length')])
        with self.assertRaises(ValueError):
            adapted(evaluation_strategy='steps', eval_strategy='epoch', max_seq_length=2048, untouched=10)
        def old_api(*, tokenizer):
            return tokenizer
        old_patches = []
        old = baseline.adapt_keywords(old_api, {'tokenizer': 'processing_class'}, old_patches, 'SFTTrainer')
        self.assertEqual(old(tokenizer='same'), 'same')
        self.assertEqual(old_patches, [])

    def test_simple_ast_derivative_changes_only_literal_prompt_fragments(self):
        source = (baseline.VENDOR / 'eval_finetuned_models.py').read_text()
        derivative = ast.parse(baseline.derive_simple_inference(source))
        for node in ast.walk(derivative):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for old, new in baseline.PROMPT_REPLACEMENTS:
                    node.value = node.value.replace(new, old)
        self.assertEqual(ast.dump(derivative), ast.dump(ast.parse(source)))
        with self.assertRaises(ValueError):
            baseline.derive_simple_inference('def format_sample(example):\n    prompt = "changed"\n')

    def test_reasoning_order_raw_tags_and_simple_completion_preserved(self):
        module = self.trainer_module()
        config = {**module.DEFAULT_CONFIG, 'output_dir': str(self.directory), 'base_image_path': str(self.directory)}
        reason_type = baseline.trainer_class(module, 'reason', [])
        self.assertIs(reason_type, module.QwenVLTrainer)
        reason = reason_type(config)._format_sample(self.row)
        self.assertEqual(reason['messages'][1]['content'][0]['text'],
                         self.row['thinking'].strip() + '\n\n' + self.row['solution'].strip())
        simple = baseline.trainer_class(module, 'simple', [])(config)._format_sample(self.row)
        self.assertEqual(simple['messages'][1]['content'][0]['text'], self.row['solution'].strip())
        self.assertEqual(reason['messages'][0]['content'][:-1], simple['messages'][0]['content'][:-1])
        self.assertIn(str(self.camera2), simple['messages'][0]['content'][0]['image'])
        for result in (reason, simple):
            text = result['messages'][0]['content'][-1]['text']
            self.assertNotIn('PRIVATE_REFERENCE', text)
            self.assertNotIn('RAW explanation', text)
            self.assertIn(self.row['problem'], text)
        self.assertNotIn('reasoning', simple['messages'][0]['content'][-1]['text'])

    def test_original_row_split_none_entries_length_mismatch_and_steps_survive(self):
        module = self.trainer_module()
        patches = []
        baseline.install_training_compatibility(module, patches)
        config = {**module.DEFAULT_CONFIG, 'data_path': str(self.data),
                  'output_dir': str(self.directory), 'base_image_path': str(self.directory), 'use_wandb': False}
        trainer = baseline.trainer_class(module, 'simple', patches)(config)
        trainer.prepare_data()
        self.assertEqual(module.dataset_calls, [{'test_size': 0.1, 'seed': 3407}])
        self.assertIsNone(trainer.train_dataset[1])
        fake_transformers = types.SimpleNamespace(EarlyStoppingCallback=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {'transformers': fake_transformers}):
            trainer.setup_trainer()
        self.assertEqual(trainer.trainer.data_collator['max_seq_length'], 4096)
        self.assertEqual(trainer.trainer.args['max_length'], 2048)
        self.assertEqual(trainer.trainer.args['eval_strategy'], 'steps')
        self.assertEqual(trainer.trainer.args['eval_steps'], 10)
        self.assertEqual(trainer.trainer.args['save_steps'], 50)
        self.assertEqual(trainer.trainer.args['num_train_epochs'], 3)
        self.assertEqual(trainer.trainer.callbacks, [{'early_stopping_patience': 2}])
        self.assertEqual(len(patches), 4)  # Simple plus three API keyword aliases.

    def test_inference_has_no_gt_and_retains_sampling_order_and_omitted_failures(self):
        for mode in ('simple', 'reason'):
            with self.subTest(mode=mode):
                module, prompts, calls = self.inference_module(mode)
                rows = [dict(self.row, problem='Second question'), dict(self.row, problem='First question'),
                        dict(self.row, image=[])]
                with contextlib.redirect_stdout(io.StringIO()):
                    results = module.evaluate_model('adapter', rows, str(self.directory), str(self.directory / (mode + '.json')))
                self.assertEqual([r['question'] for r in results], ['<image>\nSecond question', '<image>\nFirst question'])
                self.assertEqual([r['id'] for r in results], ['scene_frame_0', 'scene_frame_1'])
                self.assertEqual(len(calls), 2)
                for call in calls:
                    self.assertEqual(call, {'max_new_tokens': 512, 'do_sample': True, 'temperature': 0.7, 'top_p': 0.9})
                for messages in prompts:
                    self.assertEqual(len(messages), 1)
                    serialized = json.dumps(messages)
                    self.assertNotIn('PRIVATE_REFERENCE', serialized)
                    self.assertNotIn('RAW explanation', serialized)
                    self.assertEqual(messages[0]['role'], 'user')
                self.assertEqual('reasoning' in prompts[0][0]['content'][-1]['text'], mode == 'reason')

    def test_inference_runtime_metadata_and_no_overwrite(self):
        module, prompts, calls = self.inference_module('simple')
        input_path = self.directory / 'ordered.jsonl'
        input_path.write_text('\n'.join(json.dumps(row) for row in [dict(self.row, problem='B'), dict(self.row, problem='A')]))
        args = argparse.Namespace(command='infer', mode='simple', data_path=str(input_path), model_path=str(self.directory),
                                  image_root=str(self.directory), output_path=str(self.directory / 'raw.json'))
        audit = {'compatibility_patches': [], 'status': 'running'}
        old_cwd = Path.cwd()
        with patch.object(baseline, 'load_upstream', return_value=(module, baseline.verify_vendor())), \
                patch.object(baseline, 'runtime_audit', return_value=audit), contextlib.redirect_stdout(io.StringIO()):
            result = baseline.run_infer(args)
        saved = json.loads(Path(args.output_path + '.runtime.json').read_text())
        self.assertEqual((saved['status'], saved['input_rows'], saved['returned_rows']), ('completed', 2, 2))
        self.assertIsNone(saved['generation']['generation_seed'])
        self.assertEqual([r['problem'] for r in baseline.ordered_rows(input_path)], ['B', 'A'])
        self.assertEqual(old_cwd, Path.cwd())
        with self.assertRaises(FileExistsError):
            baseline.run_infer(args)

    def test_training_result_reports_actual_directory_and_keeps_loss_default(self):
        module = self.trainer_module()
        args = argparse.Namespace(command='train', mode='reason', data_path=str(self.data), model_path=str(self.directory),
                                  image_root=str(self.directory), output_dir=str(self.directory / 'run'), epochs=3)
        audit = {'compatibility_patches': [], 'status': 'running'}
        fake_transformers = types.SimpleNamespace(EarlyStoppingCallback=lambda **kwargs: kwargs)
        old_cwd = Path.cwd()
        with patch.object(baseline, 'load_upstream', return_value=(module, baseline.verify_vendor())), \
                patch.object(baseline, 'runtime_audit', return_value=audit), \
                patch.dict(sys.modules, {'transformers': fake_transformers}):
            baseline.run_train(args)
        saved = json.loads((Path(args.output_dir) / 'training_result.json').read_text())
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(saved['actual_output_dir'], str((Path(args.output_dir) / 'original_timestamp_directory').resolve()))
        self.assertFalse(saved['effective_training_config']['use_wandb'])
        self.assertEqual(saved['upstream_default_config']['seed'], 3407)
        self.assertFalse(saved['collator_train_on_responses_only_default'])
        self.assertTrue(saved['example_returned_none'])
        self.assertEqual(old_cwd, Path.cwd())

    def test_unverified_collator_default_stops_before_constructing_trainer(self):
        module = self.trainer_module()
        module.UnslothVisionDataCollator = lambda train_on_responses_only=True: None
        args = argparse.Namespace(command='train', mode='reason', data_path=str(self.data), model_path=str(self.directory),
                                  image_root=str(self.directory), output_dir=str(self.directory / 'bad_loss'), epochs=3)
        with patch.object(baseline, 'load_upstream', return_value=(module, baseline.verify_vendor())), \
                patch.object(baseline, 'runtime_audit', return_value={'compatibility_patches': []}):
            with self.assertRaisesRegex(RuntimeError, 'full-text loss'):
                baseline.run_train(args)
        saved = json.loads((Path(args.output_dir) / 'training_result.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertNotIn('actual_output_dir', saved)


if __name__ == '__main__':
    unittest.main()
