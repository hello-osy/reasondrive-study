import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data_utils import write_json
from mac_pipeline import image_signature, reusable_plan, evaluation_interval


class MacOptimizationTests(unittest.TestCase):
    def test_validation_cadence_and_final_step(self):
        interval = evaluation_interval(4199, 12)
        self.assertEqual(interval, 4200)
        self.assertEqual([i for i in range(1, 12598) if i % interval == 0 or i == 12597],
                         [4200, 8400, 12597])
        self.assertEqual(evaluation_interval(1, 12), 12)
        self.assertEqual(evaluation_interval(4199, 12, 121), 132)

    def test_cache_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.jpg'
            image.write_bytes(b'original')
            rows = [{'image': [str(image)]}]
            signature, newest = image_signature(rows, root)
            expected = {'max_seq_length': 2048, 'dataset_sha256': 'data', 'reasoning_format_version': 2}
            manifest = {'indices': {'train': [0], 'validation': [1]}}
            plan = {**expected, 'usable': manifest['indices'],
                    'lengths': {mode: {'0': 10, '1': 20} for mode in ('simple', 'reason')}}
            path = root / 'plan.json'
            write_json(path, plan)
            self.assertIsNotNone(reusable_plan(path, expected, manifest, signature, newest))
            self.assertIsNotNone(reusable_plan(path, expected, manifest, signature, newest))
            self.assertIsNone(reusable_plan(path, {**expected, 'max_seq_length': 1024}, manifest, signature, newest))
            self.assertIsNone(reusable_plan(path, expected, {'indices': {'train': [2], 'validation': [1]}}, signature, newest))
            plan['image_signature'] = signature
            legacy = dict(plan)
            legacy.pop('reasoning_format_version')
            write_json(path, legacy)
            self.assertIsNone(reusable_plan(path, expected, manifest, signature, newest))
            write_json(path, plan)
            image.write_bytes(b'changed image bytes')
            changed, newest = image_signature(rows, root)
            self.assertNotEqual(changed, signature)
            self.assertIsNone(reusable_plan(path, expected, manifest, changed, newest))


if __name__ == '__main__':
    unittest.main()
