import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import hf_auth


class HFAuthTests(unittest.TestCase):
    def test_token_is_sent_but_not_printed(self):
        output = io.StringIO()
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_test_secret'}), \
             patch.object(hf_auth, 'urlopen', return_value=io.BytesIO(b'{"name":"tester"}')) as request, \
             contextlib.redirect_stdout(output):
            self.assertEqual(hf_auth.main(), 0)
        self.assertEqual(request.call_args.args[0].get_header('Authorization'), 'Bearer hf_test_secret')
        self.assertNotIn('hf_test_secret', output.getvalue())

    def test_rejected_token_stops_without_leaking_exception(self):
        output = io.StringIO()
        error = HTTPError('https://huggingface.co/api/whoami-v2', 401, 'hf_test_secret', {}, None)
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_test_secret'}), \
             patch.object(hf_auth, 'urlopen', side_effect=error), contextlib.redirect_stderr(output):
            self.assertEqual(hf_auth.main(), 1)
        self.assertIn('401', output.getvalue())
        self.assertNotIn('hf_test_secret', output.getvalue())

    def test_noninteractive_missing_token_fails(self):
        env = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'REASONDRIVE_HF_AUTH_CHECKED')}
        result = subprocess.run(['bash', '-c', 'source scripts/hf_auth.sh; require_hf_token'],
                                cwd=ROOT, env=env, input='', capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_child_reuses_verified_token(self):
        command = '''source scripts/hf_auth.sh
python3() { echo checked; }
require_hf_token
bash -c 'source scripts/hf_auth.sh; require_hf_token; [[ $HF_TOKEN == hf_test_secret ]]'
'''
        env = dict(os.environ, HF_TOKEN='hf_test_secret')
        env.pop('REASONDRIVE_HF_AUTH_CHECKED', None)
        result = subprocess.run(['bash', '-c', command], cwd=ROOT, env=env,
                                input='', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'checked')
        self.assertNotIn('hf_test_secret', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
