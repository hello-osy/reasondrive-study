"""재구현부 회귀 검사: 누락 root 복원, 문항 보존, 생성 답변 형식."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data_utils import dataset_hash, format_sample, generated_answer, tagged_text, write_json
from evaluate import LANGUAGE_KEYS, compare, route_and_score
from graph_context import ROOT_QUESTION, build_graph_context, validate_graph_context


class GeneratedAnswerTests(unittest.TestCase):
    def test_balanced_answer_and_reasoning(self):
        self.assertEqual(generated_answer('<answer> Stop </answer>'), 'Stop')
        self.assertEqual(generated_answer('<think>See <c1,CAM_FRONT,1.0,2.0>.</think>\n<answer>No.</answer>'), 'No.')
        self.assertEqual(tagged_text('<answer>Stop</answer>', 'answer', required=True), 'Stop')
        # Source parsing/normalization is deliberately independent of generated-output policy.
        self.assertEqual(tagged_text('Stop', 'answer'), 'Stop')

    def test_malformed_generated_outputs_are_rejected(self):
        bad = ['<answer>Stop</answer><answer>Go',
               '<answer><answer>Stop</answer></answer>',
               '<think>Candidate <answer>Stop</answer> considered</think>',
               '<think>Unclosed<answer>Stop</answer>',
               '<answer>Stop</answer><answer>Go</answer>',
               '<answer>Stop', '<answer> </answer>',
               '<answer>Stop</answer></answer>', '<answer x="1">Stop</answer>',
               '<answer>Stop</answer> trailing prose', 'Stop', '']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                generated_answer(text)


class GraphContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.image = self.directory / 'image.jpg'
        self.image.touch()
        self.object = '<c1,CAM_FRONT,10.0,20.0>'
        self.root_answer = f'There is a car at {self.object}.'
        specifications = [('complete', 'perception', ROOT_QUESTION, self.root_answer, [2]),
                          ('complete', 'prediction', f'Is {self.object} a car?', 'Yes.', [0]),
                          ('wrong_first', 'perception', f'What is the moving status of object {self.object}?', 'Going ahead.', [0]),
                          ('no_perception', 'prediction', f'Is {self.object} a car?', 'Yes.', [0])]
        self.rows = []
        for frame, category, question, answer, tag in specifications:
            self.rows.append({'image': [str(self.image)], 'problem': question,
                              'solution': f'<answer>{answer}</answer>', 'thinking': 'Because.',
                              'metadata': {'scene_id': 'scene', 'frame_id': frame, 'category': category,
                                           'original_qa_pair': {'Q': question, 'A': answer, 'tag': tag}}})
        self.manifest = {'dataset_sha256': dataset_hash(self.rows),
                         'indices': {'test': list(range(len(self.rows)))}}
        self.annotations = {'scene': {'key_frames': {
            frame: {'QA': {'perception': [{'Q': ROOT_QUESTION, 'A': self.root_answer}]},
                    'image_paths': {'CAM_FRONT': str(self.image)}}
            for frame in ('wrong_first', 'no_perception')}}}
        self.context = build_graph_context(self.rows, self.manifest, self.annotations,
                                           {'sha256': 'unit-test-original-annotation'})
        self.responses = [self.response(str(i), row) for i, row in enumerate(self.rows)]
        self.sidecar = {'graph_context': self.context,
                        'responses': [self.response(item['id'], item, scored=False)
                                      for item in self.context['items']]}

    @staticmethod
    def response(identifier, row, **extra):
        return {'id': identifier, 'question': row['problem'],
                'gt_answer': tagged_text(row['solution'], 'answer'),
                'answer': row['solution'], 'status': 'ok', **extra}

    @staticmethod
    def language(predictions, references):
        return {key: 1.0 for key in LANGUAGE_KEYS}

    def test_oracle_restores_both_missing_root_cases_without_new_scored_rows(self):
        with self.assertRaisesRegex(ValueError, 'separately inferred'):
            route_and_score(self.responses, self.rows, self.language)
        scores, counts = route_and_score(self.responses, self.rows, self.language, self.sidecar)
        self.assertEqual(scores['Accuracy'], 1.0)
        self.assertEqual(counts['total'], 4)
        self.assertEqual(counts['accuracy'], 3)
        self.assertEqual(counts['language'], 1)  # Context roots do not become language-score items.
        self.assertEqual(counts['context_only'], 2)
        self.assertEqual(counts['graph_excluded'], 0)

    def test_context_inference_prompt_has_no_ground_truth(self):
        for item in self.context['items']:
            prompt = format_sample(item, self.directory, 'reason', training=False)
            self.assertNotIn(self.root_answer, str(prompt))
            self.assertNotIn(self.object, str(prompt))
            self.assertEqual(len(prompt['messages']), 1)

    def test_context_errors_remain_visible_and_do_not_drop_scored_responses(self):
        sidecar = copy.deepcopy(self.sidecar)
        sidecar['responses'][0].update(status='error', answer='')
        _, counts = route_and_score(self.responses, self.rows, self.language, sidecar)
        self.assertEqual(counts['total'], 4)
        self.assertEqual(counts['context_failures'], 1)
        self.assertEqual(counts['graph_excluded'], 1)
        sidecar['responses'][0].update(status='ok', answer='<answer>Missing close')
        _, counts = route_and_score(self.responses, self.rows, self.language, sidecar)
        self.assertEqual(counts['context_failures'], 1)
        sidecar['responses'].pop()
        with self.assertRaisesRegex(ValueError, 'every restored root'):
            route_and_score(self.responses, self.rows, self.language, sidecar)

    def test_original_source_and_context_integrity_are_checked(self):
        original = copy.deepcopy(self.annotations)
        original['scene']['key_frames']['wrong_first']['image_paths']['CAM_FRONT'] = '/different.jpg'
        with self.assertRaisesRegex(ValueError, 'image identity'):
            build_graph_context(self.rows, self.manifest, original, {'sha256': 'fixture'})
        tampered = copy.deepcopy(self.context)
        tampered['items'][0]['solution'] = '<answer>Wrong</answer>'
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            validate_graph_context(tampered, self.rows, self.manifest['indices']['test'])

    def test_comparison_requires_matching_embedded_context(self):
        path = self.directory / 'responses.json'
        write_json(path, self.responses)
        write_json(str(path) + '.context.json', self.sidecar)
        write_json(str(path) + '.meta.json', {'dataset_sha256': dataset_hash(self.rows),
                   'test_indices': self.manifest['indices']['test'], 'split': 'test', 'seed': 3407,
                   'max_new_tokens': 512, 'do_sample': False,
                   'graph_context_sha256': self.context['sha256']})
        table, counts = compare([('simple', path)], self.rows, self.language)
        self.assertEqual(table[0]['Accuracy'], 1.0)
        self.assertEqual(counts['simple']['total'], 4)
        Path(str(path) + '.context.json').unlink()
        with self.assertRaisesRegex(ValueError, 'sidecar'):
            compare([('simple', path)], self.rows, self.language)


if __name__ == '__main__':
    unittest.main()
