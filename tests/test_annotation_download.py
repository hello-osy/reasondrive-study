"""Authenticated annotation download and integrity regression checks."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import graph_context as graph


class AnnotationDownloadTests(unittest.TestCase):
    def test_hub_download_is_verified_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'hub.json'
            source.write_bytes(b'{"fixture": true}')
            target = Path(directory) / 'run' / graph.ANNOTATION_NAME
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(graph, 'ANNOTATION_SHA256', digest), \
                 patch('huggingface_hub.hf_hub_download', return_value=str(source)) as hub, \
                 patch.object(graph, '_download') as mirror:
                self.assertEqual(graph.official_annotations(target), target)
                self.assertEqual(graph.official_annotations(target), target)
                hub.assert_called_once_with('OpenDriveLab/DriveLM', graph.ANNOTATION_NAME,
                                            repo_type='dataset', revision=graph.ANNOTATION_REVISION)
                mirror.assert_not_called()
                self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_corrupt_hub_file_falls_back_to_verified_mirror(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'hub.json'
            source.write_bytes(b'corrupt')
            target = Path(directory) / graph.ANNOTATION_NAME
            valid = b'{"fixture": true}'
            def download(url, output):
                Path(output).write_bytes(valid)
            with patch.object(graph, 'ANNOTATION_SHA256', hashlib.sha256(valid).hexdigest()), \
                 patch('huggingface_hub.hf_hub_download', return_value=str(source)), \
                 patch.object(graph, '_download', side_effect=download) as mirror:
                graph.official_annotations(target)
                mirror.assert_called_once()
                self.assertEqual(target.read_bytes(), valid)
                self.assertFalse(target.with_suffix('.download').exists())

    def test_failed_download_does_not_publish_bad_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / graph.ANNOTATION_NAME
            with patch('huggingface_hub.hf_hub_download', side_effect=OSError('unavailable')), \
                 patch.object(graph, '_download', side_effect=OSError('mirror unavailable')):
                with self.assertRaisesRegex(ValueError, 'Cannot restore graph roots'):
                    graph.official_annotations(target)
                self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
