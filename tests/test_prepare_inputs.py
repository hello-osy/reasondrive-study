import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data_utils import load_split, write_json
from prepare_inputs import prepare_inputs


class InputRepairTests(unittest.TestCase):
    def test_repair_repeatability_and_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'a+b.jpg'
            image.touch()  # Image integrity belongs to download_data's preceding check.
            rows = [{'metadata': {'scene_id': str(i)}, 'image': [str(image)],
                     'problem': 'Where?', 'thinking': '<think>Here',
                     'solution': '<answer>Front</answer>'} for i in range(20)]
            data, split, report = root/'data.json', root/'split.json', root/'report.json'
            write_json(data, rows)
            original = data.read_bytes()
            first = prepare_inputs(data, root, split, report)
            self.assertEqual(first, prepare_inputs(data, root, split, report))
            self.assertEqual(data.read_bytes(), original)
            self.assertEqual(len(first['repaired']), 18)
            self.assertEqual(first['errors'], [])
            self.assertEqual(first['counts'], {'test': 2, 'validation': 2, 'train': 16})
            # Missing input must produce an actionable report, not silently skip.
            image.unlink()
            with self.assertRaises(ValueError):
                prepare_inputs(data, root, split, report)
            self.assertEqual(len(json.loads(report.read_text())['errors']), 20)


if __name__ == '__main__':
    unittest.main()
