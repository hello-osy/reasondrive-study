# ============================================
# 실험용 확장·설정 변경: 공개 조건의 명시적 평가와 noGPT 표 계약 검증
# 모델·유료 API·실제 언어 평가 backend를 실행하지 않는 작은 합성 fixture.
# ============================================
import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data_utils import dataset_hash, write_json
from evaluate import LANGUAGE_KEYS, compare, public_baseline_manifest, write_report
from graph_context import ROOT_QUESTION


class ReportScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Every selected frame has its root, but any 10% QA subset is a partial scene.
        self.data = [{'problem': ROOT_QUESTION, 'solution': '<answer>Stop</answer>',
                      'metadata': {'scene_id': 'one-scene', 'frame_id': str(i),
                                   'category': 'perception',
                                   'original_qa_pair': {'Q': ROOT_QUESTION, 'A': 'Stop', 'tag': [0, 2]}}}
                     for i in range(40)]
        self.manifest = public_baseline_manifest(self.data)

    @staticmethod
    def language(predictions, references):
        return {key: 0.0 for key in LANGUAGE_KEYS}

    def result(self, variant='reason'):
        path = self.root / f'{variant}.json'
        responses = [{'id': str(i), 'question': ROOT_QUESTION, 'gt_answer': 'Stop',
                      'answer': '<answer>Stop</answer>', 'status': 'ok'}
                     for i in self.manifest['test_indices']]
        meta = {'schema_version': 2, 'protocol': 'public_baseline',
                'protocol_sha256': self.manifest['sha256'], 'variant': variant,
                'dataset_sha256': dataset_hash(self.data), 'test_indices': self.manifest['test_indices'],
                'split': 'test', 'seed': None, 'max_new_tokens': 512, 'do_sample': True,
                'temperature': 0.7, 'top_p': 0.9, 'backend': 'cuda', 'image_size': 224,
                'experiment_sha256': 'same-public-experiment',
                'model_source': {'model_id': 'fixture-base', 'resolved_revision': 'fixture-revision',
                                 'processor_revision': 'fixture-revision'}}
        write_json(path, responses)
        write_json(str(path) + '.meta.json', meta)
        write_json(str(path) + '.protocol.json', self.manifest)
        return path, responses, meta

    def test_partial_scene_requires_explicit_public_protocol(self):
        path, _, meta = self.result()
        with self.assertRaisesRegex(ValueError, 'protocol'):
            compare([('reason', path)], self.data, self.language)
        table, counts = compare([('reason', path)], self.data, self.language, protocol='public_baseline')
        self.assertEqual(table[0]['Accuracy'], 1.0)
        self.assertEqual(counts['reason']['total'], 4)
        self.assertEqual(counts['reason']['training_overlap_scenes'], 1)
        # Even without protocol metadata, the original C1 guard still rejects this subset.
        meta.pop('protocol')
        write_json(str(path) + '.meta.json', meta)
        with self.assertRaisesRegex(ValueError, 'Incomplete test scenes'):
            compare([('reason', path)], self.data, self.language)

    def test_simple_reason_share_manifest_and_conditions(self):
        simple, _, _ = self.result('simple')
        reason, _, _ = self.result('reason')
        table, _ = compare([('simple', simple), ('reason', reason)], self.data, self.language,
                           protocol='public_baseline')
        self.assertEqual([row['model'] for row in table], ['simple', 'reason'])
        with self.assertRaisesRegex(ValueError, 'Duplicate public baseline model variant'):
            compare([('first', reason), ('second', reason)], self.data, self.language,
                    protocol='public_baseline')

    def test_rehashed_but_changed_upstream_partitions_are_rejected(self):
        path, _, meta = self.result()
        for field in ('train_seed', 'eval_seed', 'upstream_commit', 'training_train_indices',
                      'training_eval_indices', 'test_indices'):
            changed = copy.deepcopy(self.manifest)
            if isinstance(changed[field], list):
                changed[field][0], changed[field][1] = changed[field][1], changed[field][0]
            else:
                changed[field] = 'tampered'
            changed['sha256'] = dataset_hash({key: value for key, value in changed.items() if key != 'sha256'})
            write_json(str(path) + '.protocol.json', changed)
            write_json(str(path) + '.meta.json', dict(meta, protocol_sha256=changed['sha256']))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'pinned upstream splits'):
                compare([('reason', path)], self.data, self.language, protocol='public_baseline')

    def test_missing_manifest_or_incomplete_metadata_rejected(self):
        path, _, meta = self.result()
        Path(str(path) + '.protocol.json').unlink()
        with self.assertRaisesRegex(ValueError, 'manifest required'):
            compare([('reason', path)], self.data, self.language, protocol='public_baseline')
        write_json(str(path) + '.protocol.json', self.manifest)
        for field in ('schema_version', 'protocol', 'protocol_sha256', 'variant', 'model_source'):
            changed = dict(meta)
            changed.pop(field)
            write_json(str(path) + '.meta.json', changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare([('reason', path)], self.data, self.language, protocol='public_baseline')

    def test_changed_sampling_or_eval_order_rejected(self):
        path, _, meta = self.result()
        for field, value in [('do_sample', False), ('temperature', 0.0), ('top_p', 1.0),
                             ('max_new_tokens', 256), ('test_indices', list(reversed(meta['test_indices'])))]:
            write_json(str(path) + '.meta.json', dict(meta, **{field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare([('reason', path)], self.data, self.language, protocol='public_baseline')

    def test_all_failed_rows_remain_in_eligible_denominator(self):
        path, responses, _ = self.result()
        responses[1].update(status='error', answer='<answer>Stop</answer>')
        responses[2].update(status='invalid_format', answer='Stop')
        responses[3].update(status='ok', answer='<answer>Stop', finish_reason='length')
        write_json(path, responses)
        table, counts = compare([('reason', path)], self.data, self.language, protocol='public_baseline')
        self.assertEqual(table[0]['Accuracy'], 0.25)
        self.assertEqual(counts['reason']['accuracy'], 4)
        self.assertEqual(counts['reason']['language'], 4)
        self.assertEqual(counts['reason']['status_ok'], 2)
        self.assertEqual(counts['reason']['status_error'], 1)
        self.assertEqual(counts['reason']['status_invalid_format'], 1)
        self.assertEqual(counts['reason']['invalid_format'], 2)
        self.assertEqual(counts['reason']['valid_answers'], 1)
        self.assertEqual(counts['reason']['empty_predictions'], 3)
        self.assertEqual(counts['reason']['finish_length'], 1)
        write_json(path, responses[:-1])
        with self.assertRaisesRegex(ValueError, 'every held-out test row'):
            compare([('reason', path)], self.data, self.language, protocol='public_baseline')

    def test_report_contains_scores_counts_scope_and_no_delta(self):
        path, _, _ = self.result()
        table, counts = compare([('reason', path)], self.data, self.language, protocol='public_baseline')
        output = self.root / 'report'
        with contextlib.redirect_stdout(io.StringIO()):
            report = write_report(table, counts, output, protocol='public_baseline',
                                  provenance={'fixture': True})
        saved = json.loads((output / 'comparison.json').read_text())
        self.assertEqual(saved, report)
        self.assertEqual(saved['rows'], table)
        self.assertIn('not an independent', saved['condition_scope'])
        self.assertIn('differ from the original paper', saved['metric_scope'])
        with (output / 'comparison.csv').open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['N_accuracy'], '4')
        self.assertEqual(row['total'], '4')
        self.assertEqual(row['valid_answers'], '4')
        self.assertEqual(row['Final_noGPT'], 'N/A')
        self.assertNotIn('delta', ' '.join(row).lower())
        markdown = (output / 'comparison.md').read_text()
        self.assertIn('status_error', markdown)
        self.assertIn('No GPT calls', markdown)


if __name__ == '__main__':
    unittest.main()
