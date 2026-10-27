"""[논문 기반 재구현] six-view/GT 기반 생성 계약 검증.

[실험용 확장·설정 변경] provider 설정·재시도·재개·dry-run은 mock만 사용한다.
이 테스트는 유료 API 호출, 모델 품질 또는 원논문 데이터 동일성을 검증하지 않는다.
"""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generate_reasoning import CAMERAS, CATEGORY_PROMPTS, digest, prompts, run, valid_thinking
from reasoning_providers import ProviderConfig, ProviderError, build_request, generate, response_result, http_transport


def openai_response(text='<think>A pedestrian is ahead, so stop.</think>', **changes):
    result = {'id': 'resp_mock', 'model': 'gpt-4o-snapshot-mock', 'status': 'completed',
              'usage': {'input_tokens': 123, 'output_tokens': 14},
              'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]}
    result.update(changes)
    return result


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.images = [{'camera': camera, 'data': b'actual-' + camera.encode(), 'mime_type': 'image/jpeg'}
                       for camera in CAMERAS]
        self.keys = patch.dict(os.environ, {'OPENAI_API_KEY': 'mock-secret', 'GEMINI_API_KEY': 'mock-gemini'})
        self.keys.start()
        self.addCleanup(self.keys.stop)

    def test_openai_request_contains_six_image_bytes_and_prompts(self):
        config = ProviderConfig()
        url, payload = build_request(config, 'SYSTEM', 'QUESTION and GROUND TRUTH', self.images)
        self.assertEqual(url, 'https://api.openai.com/v1/responses')
        content = payload['input'][0]['content']
        self.assertEqual(sum(part['type'] == 'input_image' for part in content), 6)
        self.assertEqual(payload['instructions'], 'SYSTEM')
        self.assertIn('GROUND TRUTH', content[0]['text'])
        self.assertTrue(content[2]['image_url'].startswith('data:image/jpeg;base64,'))
        self.assertNotIn('temperature', payload)
        self.assertFalse(payload['store'])
        captured = Mock(return_value=openai_response())
        result = generate(config, 'SYSTEM', 'USER', self.images, transport=captured)
        self.assertEqual(captured.call_args.args[2]['Authorization'], 'Bearer mock-secret')
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(result['usage']['output_tokens'], 14)

    def test_gemini_request_and_visible_text_response(self):
        config = ProviderConfig(provider='gemini', model='gemini-test', temperature=0.3)
        url, payload = build_request(config, 'SYSTEM', 'USER', self.images)
        self.assertEqual(url, 'https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent')
        self.assertNotIn('key=', url)
        self.assertEqual(payload['systemInstruction']['parts'][0]['text'], 'SYSTEM')
        self.assertEqual(sum('inlineData' in part for part in payload['contents'][0]['parts']), 6)
        self.assertEqual(payload['generationConfig']['temperature'], 0.3)
        response = {'responseId': 'gemini_mock', 'modelVersion': 'gemini-test-snapshot',
                    'usageMetadata': {'totalTokenCount': 22}, 'candidates': [{'finishReason': 'STOP',
                     'content': {'parts': [{'thought': True, 'text': 'internal'},
                                           {'text': '<think>Visible explanation.</think>'}]}}]}
        transport = Mock(return_value=response)
        result = generate(config, 'SYSTEM', 'USER', self.images, transport=transport)
        self.assertEqual(transport.call_args.args[2]['x-goog-api-key'], 'mock-gemini')
        self.assertEqual(result['text'], '<think>Visible explanation.</think>')
        self.assertTrue(result['complete'])
        self.assertFalse(response_result(config, {'candidates': []})['complete'])

    def test_retry_transient_errors_only_and_bounded_attempts(self):
        transport = Mock(side_effect=[ProviderError('429', retryable=True),
                                      ProviderError('503', retryable=True), openai_response()])
        sleep = Mock()
        result = generate(ProviderConfig(), 'S', 'U', self.images, transport=transport, sleep=sleep)
        self.assertEqual(result['attempts'], 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        transport = Mock(side_effect=ProviderError('401', retryable=False))
        with self.assertRaises(ProviderError) as caught:
            generate(ProviderConfig(), 'S', 'U', self.images, transport=transport, sleep=sleep)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(caught.exception.attempts, 1)
        transport = Mock(side_effect=ProviderError('503', retryable=True))
        with self.assertRaises(ProviderError) as caught:
            generate(ProviderConfig(max_attempts=2), 'S', 'U', self.images, transport=transport, sleep=Mock())
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(caught.exception.attempts, 2)

    def test_payload_limit_no_network_and_config_validation(self):
        transport = Mock()
        with self.assertRaisesRegex(ProviderError, 'exceeds'):
            generate(ProviderConfig(max_request_bytes=10), 'S', 'U', self.images, transport=transport)
        transport.assert_not_called()
        for kwargs in ({'max_attempts': 0}, {'temperature': float('nan')}, {'provider': 'gemini'},
                       {'api_key_env': 'not a name'}, {'timeout_seconds': -1}, {'model': '../x'},
                       {'max_output_tokens': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ProviderConfig(**kwargs)

    def test_incomplete_and_refusal_are_not_successful(self):
        self.assertFalse(response_result(ProviderConfig(), openai_response(status='incomplete'))['complete'])
        refusal = openai_response(output=[{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'no'}]}])
        self.assertFalse(response_result(ProviderConfig(), refusal)['complete'])

    def test_http_error_does_not_expose_response_body_or_key(self):
        from urllib.error import HTTPError
        import io
        response = HTTPError('https://api.openai.com/v1/responses', 401, 'mock-secret', {},
                             io.BytesIO(b'mock-secret raw API error'))
        with patch('reasoning_providers.request.urlopen', side_effect=response), self.assertRaises(ProviderError) as caught:
            http_transport('https://api.openai.com/v1/responses', {}, {'Authorization': 'Bearer mock-secret'}, 1)
        self.assertNotIn('mock-secret', str(caught.exception))
        self.assertEqual(caught.exception.status, 401)


class DatasetGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.json'
        self.output = self.root / 'generated.json'
        self.images = []
        for camera in CAMERAS:
            path = self.root / 'nuscenes' / 'samples' / camera / 'frame.jpg'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'\xff\xd8\xff' + camera.encode())
            self.images.append('../nuscenes/samples/' + camera + '/frame.jpg')
        self.rows = [{'image': self.images, 'problem': 'What should the ego vehicle do?',
                      'solution': '<answer>Stop.</answer>', 'thinking': 'old teacher text',
                      'metadata': {'scene_id': 'scene-1', 'frame_id': 'frame-1', 'category': category,
                                   'original_qa_pair': {'Q': 'What should the ego vehicle do?', 'A': 'Stop.', 'tag': [1]}}}
                     for category in ('perception', 'planning')]
        self.source.write_text(json.dumps(self.rows))
        self.kwargs = {'data_path': self.source, 'output_path': self.output,
                       'image_root': self.root / 'nuscenes', 'config': ProviderConfig()}
        self.keys = patch.dict(os.environ, {'OPENAI_API_KEY': 'mock-secret'})
        self.keys.start()
        self.addCleanup(self.keys.stop)

    @staticmethod
    def successful(config, system, user, images):
        return dict(response_result(config, openai_response()), attempts=1)

    def test_all_category_prompts_and_ground_truth_are_present(self):
        for category in CATEGORY_PROMPTS:
            row = copy.deepcopy(self.rows[0])
            row['metadata']['category'] = category
            system, user = prompts(row)
            self.assertIn(category, system)
            self.assertIn(row['problem'], user)
            self.assertIn('Ground-truth answer: Stop.', user)
            self.assertNotIn('old teacher text', user)
        bad = copy.deepcopy(self.rows[0])
        bad['metadata']['original_qa_pair']['A'] = 'Go.'
        with self.assertRaises(ValueError):
            prompts(bad)

    def test_dry_run_requires_no_key_and_does_not_call_generator(self):
        generator = Mock(side_effect=AssertionError('No API calls in dry-run'))
        with patch.dict(os.environ, {}, clear=True):
            summary = run(**self.kwargs, generator=generator)
        self.assertEqual(summary['valid'], 2)
        self.assertEqual(summary['api_calls'], 0)
        self.assertTrue(Path(str(self.output) + '.plan.json').exists())
        self.assertFalse(self.output.exists())
        generator.assert_not_called()

    def test_generation_preserves_truth_metadata_and_reuses_completed_rows(self):
        original = self.source.read_bytes()
        generator = Mock(side_effect=self.successful)
        summary = run(**self.kwargs, execute=True, limit=1, generator=generator)
        self.assertFalse(summary['complete'])
        generated = json.loads(self.output.read_text())
        self.assertEqual(generated[1]['thinking'], '')
        self.assertEqual(generated[1]['reasoning_provenance']['status'], 'pending')
        first = generated[0]
        summary = run(**self.kwargs, execute=True, resume=True, generator=generator)
        self.assertTrue(summary['complete'])
        self.assertEqual(generator.call_count, 2)
        generated = json.loads(self.output.read_text())
        self.assertEqual(first, generated[0])
        for source, result in zip(self.rows, generated):
            for key in ('solution', 'metadata', 'image', 'problem'):
                self.assertEqual(source[key], result[key])
            self.assertEqual(result['reasoning_provenance']['status'], 'ok')
            self.assertEqual(len(result['reasoning_provenance']['images']), 6)
        self.assertEqual(self.source.read_bytes(), original)
        run(**self.kwargs, execute=True, resume=True, generator=generator)
        self.assertEqual(generator.call_count, 2)
        self.assertNotIn('mock-secret', self.output.read_text())

    def test_resume_rejects_changed_model_data_and_image(self):
        run(**self.kwargs, execute=True, limit=1, generator=self.successful)
        with self.assertRaisesRegex(ValueError, 'Resume requires'):
            run(**dict(self.kwargs, config=replace(ProviderConfig(), model='gpt-4o-other')),
                execute=True, resume=True, generator=self.successful)
        self.rows[0]['thinking'] = 'changed source'
        self.source.write_text(json.dumps(self.rows))
        with self.assertRaisesRegex(ValueError, 'Resume requires'):
            run(**self.kwargs, execute=True, resume=True, generator=self.successful)
        self.rows[0]['thinking'] = 'old teacher text'
        self.source.write_text(json.dumps(self.rows))
        image = self.root / 'nuscenes' / 'samples' / CAMERAS[0] / 'frame.jpg'
        image.write_bytes(image.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Resume requires'):
            run(**self.kwargs, execute=True, resume=True, generator=self.successful)

    def test_failed_format_is_empty_and_requires_explicit_retry(self):
        bad = Mock(return_value=dict(response_result(ProviderConfig(), openai_response('unclosed <think>')), attempts=1))
        summary = run(**self.kwargs, execute=True, generator=bad)
        self.assertEqual(summary['failed'], 2)
        result = json.loads(self.output.read_text())
        self.assertEqual(result[0]['thinking'], '')
        self.assertEqual(result[0]['reasoning_provenance']['raw_text'], 'unclosed <think>')
        good = Mock(side_effect=self.successful)
        run(**self.kwargs, execute=True, resume=True, generator=good)
        good.assert_not_called()
        summary = run(**self.kwargs, execute=True, resume=True, retry_failed=True, generator=good)
        self.assertTrue(summary['complete'])
        self.assertEqual(json.loads(self.output.read_text())[0]['reasoning_provenance']['row_attempt'], 2)

    def test_interrupted_request_is_not_silently_submitted_again(self):
        interrupted = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            run(**self.kwargs, execute=True, limit=1, generator=interrupted)
        good = Mock(side_effect=self.successful)
        summary = run(**self.kwargs, execute=True, resume=True, generator=good)
        self.assertEqual(summary['started'], 1)
        self.assertEqual(good.call_count, 1)
        summary = run(**self.kwargs, execute=True, resume=True, retry_failed=True, generator=good)
        self.assertTrue(summary['complete'])

    def test_input_and_existing_output_are_protected(self):
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            run(**dict(self.kwargs, output_path=self.source))
        sidecar_source = Path(str(self.output) + '.plan.json')
        sidecar_source.write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            run(**dict(self.kwargs, data_path=sidecar_source))
        run(**self.kwargs, execute=True, generator=self.successful)
        with self.assertRaises(FileExistsError):
            run(**self.kwargs, execute=True, generator=self.successful)
        self.assertEqual(self.source.read_bytes(), original)

    def test_authentication_failure_stops_remaining_requests(self):
        generator = Mock(side_effect=ProviderError('API returned HTTP 401', status=401))
        summary = run(**self.kwargs, execute=True, generator=generator)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['pending'], 1)
        self.assertEqual(generator.call_count, 1)

    def test_concurrent_execution_is_rejected_before_any_api_request(self):
        import fcntl
        with Path(str(self.output) + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            generator = Mock()
            with self.assertRaisesRegex(ValueError, 'Another process'):
                run(**self.kwargs, execute=True, generator=generator)
            generator.assert_not_called()

    def test_missing_camera_is_failed_without_api_and_old_thinking_is_not_retained(self):
        self.rows[0]['image'] = self.images[:-1]
        self.source.write_text(json.dumps(self.rows))
        generator = Mock(side_effect=self.successful)
        summary = run(**self.kwargs, execute=True, generator=generator)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(generator.call_count, 1)
        result = json.loads(self.output.read_text())
        self.assertEqual(result[0]['thinking'], '')
        self.assertEqual(result[0]['reasoning_provenance']['attempts'], 0)

    def test_completed_response_with_answer_or_extra_text_is_rejected(self):
        for text in ('', '<think></think>', '<think>first</think><think>second</think>',
                     '<think>x<answer>Stop</answer></think>', 'Preface <think>x</think>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                valid_thinking(text)
        self.assertEqual(valid_thinking(' <think> Valid. </think> '), '<think>Valid.</think>')


if __name__ == '__main__':
    unittest.main()
