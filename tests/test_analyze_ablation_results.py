"""CPU regression tests for leakage-free baseline fitting and ID-based pairing."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_ablation_results import fit_category_majority, paired_transitions


def row(answer):
    return {'solution': f'<answer>{answer}</answer>',
            'metadata': {'category': 'behavior', 'original_qa_pair': {'tag': [0]}}}


def scored(correct, truth='A'):
    return {'correct': correct, 'truth': truth, 'category': 'behavior', 'stop': False}


class AnalysisTests(unittest.TestCase):
    def test_majority_uses_training_rows_only(self):
        rows = [row('A'), row('A'), row('B'), *[row('B') for _ in range(100)]]
        fitted = fit_category_majority(rows, [0, 1, 2])
        self.assertEqual(fitted['behavior']['answer'], 'A')
        self.assertEqual(fitted['behavior']['train_tag0_count'], 3)
        # Changes outside training cannot affect the fitted baseline.
        self.assertEqual(fitted, fit_category_majority(rows[:3] + [row('C')] * 200, [0, 1, 2]))

    def test_pairs_align_ids_not_file_order(self):
        before = {'1': scored(True), '2': scored(False), '3': scored(True)}
        after = {'3': scored(False), '1': scored(True), '2': scored(True)}
        result = paired_transitions(before, after)['behavior']
        self.assertEqual(result['both_correct'], 1)
        self.assertEqual(result['wrong_to_correct'], 1)
        self.assertEqual(result['correct_to_wrong'], 1)
        self.assertEqual(result['accuracy_delta'], 0)

    def test_pairs_reject_mismatched_ids_or_references(self):
        with self.assertRaisesRegex(ValueError, 'identical row IDs'):
            paired_transitions({'1': scored(True)}, {'2': scored(True)})
        with self.assertRaisesRegex(ValueError, 'references'):
            paired_transitions({'1': scored(True)}, {'1': scored(True, 'B')})


if __name__ == '__main__':
    unittest.main()
