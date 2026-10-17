import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data_utils import load_split, format_sample, write_json, dataset_hash
from evaluate import compare, route_and_score, final_no_gpt, LANGUAGE_KEYS


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / 'image.jpg'
        self.image.touch()
        self.rows = [{'metadata': {'scene_id': str(i//2)}, 'image': [str(self.image)],
                      'problem': f'Question {i}', 'thinking': '<think>Because</think>',
                      'solution': '<answer>Stop</answer>'} for i in range(40)]
        self.path = self.root / 'splits' / 'split.json'

    def test_scene_disjoint_and_reusable(self):
        split = load_split(self.rows, self.path, create=True)
        self.assertEqual(split, load_split(self.rows, self.path))
        groups = [{self.rows[i]['metadata']['scene_id'] for i in split['indices'][name]}
                  for name in ('train', 'validation', 'test')]
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])

    def test_changed_data_or_seed_rejected(self):
        load_split(self.rows, self.path, create=True)
        with self.assertRaises(ValueError):
            load_split(self.rows, self.path, seed=3047)
        self.rows[0]['problem'] = 'changed'
        with self.assertRaises(ValueError):
            load_split(self.rows, self.path)

    def test_tampered_split_rejected(self):
        split = load_split(self.rows, self.path, create=True)
        split['indices']['test'].append(split['indices']['train'][0])
        write_json(self.path, split)
        with self.assertRaises(ValueError):
            load_split(self.rows, self.path)

    def test_modes_and_no_truth_in_inference(self):
        simple = format_sample(self.rows[0], self.root, 'simple')
        reason = format_sample(self.rows[0], self.root, 'reason')
        self.assertEqual(simple['messages'][1]['content'][0]['text'], '<answer>Stop</answer>')
        self.assertIn('<think>Because</think>', reason['messages'][1]['content'][0]['text'])
        inference = format_sample(self.rows[0], self.root, 'reason', training=False)
        self.assertEqual(len(inference['messages']), 1)
        self.assertNotIn('Because', str(inference))

    def test_missing_image_and_reason_rejected_for_both_modes(self):
        for mode in ('simple', 'reason'):
            row = dict(self.rows[0], thinking='')
            with self.assertRaises(ValueError):
                format_sample(row, self.root, mode)
            row = dict(self.rows[0], image=['missing.jpg'])
            with self.assertRaises(FileNotFoundError):
                format_sample(row, self.root, mode)

    def test_training_preparation_filters_failures_and_keeps_test_untouched(self):
        import ast
        import logging
        import types
        source = Path(__file__).resolve().parents[1] / 'scripts' / 'vlm-finetuning.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_data')
        module = ast.Module(body=[method], type_ignores=[])
        data = self.root / 'data.json'
        write_json(data, self.rows)
        split = load_split(self.rows, self.path, create=True)
        # Select a bad train row and preserve independent test rows.
        bad = split['indices']['train'][0]
        self.rows[bad]['image'] = ['missing.jpg']
        self.path.unlink()
        write_json(data, self.rows)
        scope = dict(os=__import__('os'), logger=logging.getLogger('test'),
                     Dataset=types.SimpleNamespace(from_json=lambda p: json.loads(Path(p).read_text()),
                                                  from_list=lambda rows: rows),
                     load_split=load_split, format_sample=format_sample, write_json=write_json)
        exec(compile(module, str(source), 'exec'), scope)
        trainer = types.SimpleNamespace(config={'data_path': str(data), 'split_path': str(self.path),
                      'seed': 3407, 'val_size': 0.1, 'test_size': 0.1,
                      'base_image_path': str(self.root), 'mode': 'reason'}, output_dir=self.root/'run')
        scope['prepare_data'](trainer)
        self.assertEqual(len(trainer.train_dataset), len(split['indices']['train'])-1)
        self.assertTrue(all(item is not None for item in trainer.train_dataset))
        self.assertFalse(hasattr(trainer, 'test_dataset'))
        skipped = json.loads((trainer.output_dir/'skipped_samples.json').read_text())
        self.assertEqual(skipped[0]['index'], bad)

    def test_multiple_thinking_blocks_supported(self):
        row = dict(self.rows[0], thinking='<think>First</think>\n<think>Second</think>')
        formatted = format_sample(row, self.root, 'reason')
        self.assertIn('First\n\nSecond', formatted['messages'][1]['content'][0]['text'])

    def test_graph_routing_and_no_gpt_formula(self):
        questions = ['Objects?', 'Is <c1,CAM_FRONT,10.0,20.0> a car?',
                     'Is <c2,CAM_FRONT,200.0,200.0> a car?', 'Next object?', 'Safe action?']
        answers = ['<c1,CAM_FRONT,10.0,20.0>', 'Yes.', 'Yes.', '<c1,CAM_FRONT,10.0,20.0>', 'Stop']
        tags = [[2], [0], [0], [3], [1]]
        categories = ['perception', 'prediction', 'prediction', 'prediction', 'planning']
        data, responses = [], []
        for i, (q, a, t, c) in enumerate(zip(questions, answers, tags, categories)):
            data.append({'problem': q, 'solution': f'<answer>{a}</answer>',
                         'metadata': {'scene_id': 's', 'frame_id': 'f', 'category': c,
                                      'original_qa_pair': {'Q': q, 'A': a, 'tag': t}}})
            responses.append({'id': str(i), 'question': q, 'gt_answer': a,
                              'answer': f'<answer>{a}</answer>', 'status': 'ok'})
        scores, counts = route_and_score(responses, data, lambda p, r: {k: 1.0 for k in LANGUAGE_KEYS})
        self.assertEqual(scores['Accuracy'], 1.0)
        self.assertEqual(counts['graph_excluded'], 1)
        self.assertEqual(counts['gpt_only_omitted'], 1)
        self.assertAlmostEqual(scores['Match_noGPT'], 1.0, places=6)
        self.assertAlmostEqual(scores['Final_noGPT'], 0.9, places=6)
        self.assertIsNone(final_no_gpt(None, {k: 1 for k in LANGUAGE_KEYS}, 1))
        responses[1]['answer'] = '<answer>yes.</answer>'
        scores, _ = route_and_score(responses, data, lambda p, r: {k: 1 for k in LANGUAGE_KEYS})
        self.assertEqual(scores['Accuracy'], 0.0)  # official equality is case-sensitive

    def results(self):
        path = self.root / 'nested' / 'responses.json'
        self.rows = [dict(self.rows[0], metadata={'scene_id': 'a', 'frame_id': 'f', 'category': 'perception', 'original_qa_pair': {'Q': 'Q', 'A': 'Stop', 'tag': [0, 2]}}),
                     dict(self.rows[1], metadata={'scene_id': 'a', 'frame_id': 'g', 'category': 'perception', 'original_qa_pair': {'Q': 'Q2', 'A': 'Stop', 'tag': [0, 2]}})]
        self.rows[0]['problem'] = 'Q'
        self.rows[1]['problem'] = 'Q2'
        rows = [{'id': '0', 'question': 'Q', 'gt_answer': 'Stop', 'answer': '<think>Go</think><answer>Stop</answer>', 'status': 'ok'},
                {'id': '1', 'question': 'Q2', 'gt_answer': 'Stop', 'answer': '', 'status': 'error'}]
        write_json(path, rows)
        write_json(str(path)+'.meta.json', {'test_indices': [0, 1], 'dataset_sha256': dataset_hash(self.rows),
                   'split': 'test', 'seed': 3407, 'max_new_tokens': 512, 'do_sample': False})
        return path, rows

    def test_failure_denominator_and_answer_extraction(self):
        path, _ = self.results()
        result, counts = compare([('simple', path), ('reason', path)], self.rows, lambda p, r: {k: 0.0 for k in LANGUAGE_KEYS})
        self.assertEqual(result[0]['Accuracy'], 0.5)
        self.assertEqual(counts['simple']['accuracy'], 2)
        self.assertNotIn('local_token_f1', result[0])

    def test_incomplete_comparison_rejected(self):
        path, rows = self.results()
        write_json(path, rows[:1])
        with self.assertRaises(ValueError):
            compare([('simple', path)], self.rows)

    def test_different_references_rejected(self):
        path, rows = self.results()
        second = self.root / 'second.json'
        rows[0]['gt_answer'] = 'Go'
        write_json(second, rows)
        write_json(str(second)+'.meta.json', json.loads(Path(str(path)+'.meta.json').read_text()))
        with self.assertRaises(ValueError):
            compare([('simple', path), ('reason', second)], self.rows, lambda p, r: {k: 0.0 for k in LANGUAGE_KEYS})


if __name__ == '__main__':
    unittest.main()
