import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_data import archive_key, download, extract_required, required_images


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.relative = 'samples/CAM_FRONT/example.jpg'
        self.rows = [{'image': ['../nuscenes/' + self.relative]}] * 2
        self.data = self.root / 'drivelm.json'
        self.data.write_text(json.dumps(self.rows))
        image = io.BytesIO()
        Image.new('RGB', (8, 8)).save(image, format='JPEG')
        self.jpeg = image.getvalue()
        self.archive = self.root / 'images.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('nuscenes/' + self.relative, self.jpeg)
            z.writestr('../../outside.jpg', self.jpeg)
            z.writestr('nuscenes/samples/CAM_BACK/unneeded.jpg', self.jpeg)

    def test_paths_and_deduplication(self):
        self.assertEqual(required_images(self.rows), {self.relative})
        with self.assertRaises(ValueError):
            required_images([{'image': ['../nuscenes/samples/../../outside.jpg']}])
        self.assertIsNone(archive_key('../samples/CAM_FRONT/x.jpg'))
        self.assertIsNone(archive_key('/samples/CAM_FRONT/x.jpg'))

    def test_selective_extraction_and_resume(self):
        images = self.root / 'nuscenes'
        with patch('download_data.fetch_archive', return_value=self.archive) as fetch:
            report = download(self.data, images, self.root / 'cache')
            self.assertEqual(report['required_images'], 1)
            self.assertEqual(report['missing'], [])
            fetch.assert_called_once()
        self.assertFalse((images / 'samples/CAM_BACK/unneeded.jpg').exists())
        with patch('download_data.fetch_archive', side_effect=AssertionError('must reuse')):
            download(self.data, images, self.root / 'cache')
        (images / self.relative).write_bytes(b'broken')
        with patch('download_data.fetch_archive', return_value=self.archive) as fetch:
            download(self.data, images, self.root / 'cache')
            fetch.assert_called_once()
        self.assertEqual((images / self.relative).read_bytes(), self.jpeg)

    def test_invalid_payload_not_committed(self):
        bad = self.root / 'bad.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            z.writestr('nuscenes/' + self.relative, b'not an image')
        with self.assertRaises(ValueError):
            extract_required(bad, self.root / 'out', {self.relative})
        self.assertFalse((self.root / 'out' / self.relative).exists())
        self.assertFalse(list((self.root / 'out').rglob('*.part')))

    def test_missing_members_fail_with_report(self):
        self.data.write_text(json.dumps([{'image': ['../nuscenes/samples/CAM_FRONT/absent.jpg']}]))
        with patch('download_data.fetch_archive', return_value=self.archive), self.assertRaises(RuntimeError):
            download(self.data, self.root / 'out', self.root / 'cache')
        report = json.loads((self.root / 'download_report.json').read_text())
        self.assertEqual(len(report['missing']), 1)


if __name__ == '__main__':
    unittest.main()
