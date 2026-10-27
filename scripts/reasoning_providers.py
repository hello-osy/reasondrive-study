# ============================================
# [논문 기반 재구현] 파일 전체: ReasonDrive §3.1의 GPT-4o reasoning 생성에 필요한 통신 모듈.
# 공개 생성 코드가 없어 새로 작성했다. 구체적 REST 선택과 운영 설정은 논문에 명시되어 있지 않다.
# provider/model 선택, Gemini와 재시도는 아래 [실험용 확장·설정 변경] 부분에 별도로 표시한다.
# ============================================
"""[논문 기반 재구현] ReasonDrive §3.1의 이미지 기반 reasoning 생성용 API 통신.

[실험용 확장·설정 변경] provider/model 선택, Gemini, 재시도와 응답 출처 기록.
논문은 GPT-4o만 사용하며 API 버전/temperature/출력 한도는 명시하지 않았다.
이 모듈은 평가 점수를 계산하지 않는다. 키는 환경변수에서만 읽는다.

2026-10-26 확인한 공식 REST 규격:
https://developers.openai.com/api/reference/resources/responses/methods/create.md
https://developers.openai.com/api/docs/guides/images-vision
https://ai.google.dev/api/generate-content
"""
import base64
from dataclasses import asdict, dataclass
import json
import math
import os
import re
import time
from urllib import error, request


class ProviderError(RuntimeError):
    """응답 본문/인증 헤더를 노출하지 않는 API 오류."""

    def __init__(self, message, *, retryable=False, status=None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.attempts = 0


# ============================================
# [실험용 확장·설정 변경] provider/model과 명시적인 API 운영 설정.
# ============================================
@dataclass(frozen=True)
class ProviderConfig:
    # [실험용 확장·설정 변경] 논문에 없는 운영 설정을 명시적으로 저장한다.
    provider: str = 'openai'
    model: str = 'gpt-4o'
    api_key_env: str = ''
    max_output_tokens: int = 512
    temperature: float | None = None
    image_detail: str = 'auto'
    timeout_seconds: float = 90.0
    max_attempts: int = 3
    retry_base_seconds: float = 1.0
    max_request_bytes: int = 19_000_000

    def __post_init__(self):
        if self.provider not in ('openai', 'gemini'):
            raise ValueError('provider must be openai or gemini')
        if not isinstance(self.model, str) or not re.fullmatch(r'[A-Za-z0-9._:-]+', self.model):
            raise ValueError('model must be an explicit model ID, without a URL or slash')
        if self.provider == 'gemini' and self.model == 'gpt-4o':
            raise ValueError('Gemini requires an explicit --model')
        if self.api_key_env and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', self.api_key_env):
            raise ValueError('api_key_env must name an environment variable')
        for name in ('max_output_tokens', 'max_attempts', 'max_request_bytes'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('timeout_seconds', 'retry_base_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.temperature is not None and (isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ValueError('temperature must be null or a finite number from 0 to 2')
        if self.image_detail not in ('auto', 'low', 'high'):
            raise ValueError('image_detail must be auto, low or high')

    @property
    def key_name(self):
        return self.api_key_env or ('OPENAI_API_KEY' if self.provider == 'openai' else 'GEMINI_API_KEY')

    def public_dict(self):
        values = asdict(self)
        values['api_key_env'] = self.key_name
        return values


def api_key(config):
    value = os.environ.get(config.key_name, '').strip()
    if not value:
        raise ValueError(f'Set {config.key_name} before --execute; dry-run needs no key')
    return value


def build_request(config, system_prompt, user_prompt, images):
    """실제 여섯 이미지 bytes를 provider 규격의 JSON에 포함한다."""
    if len(images) != 6:
        raise ValueError('Exactly six camera images are required')
    if config.provider == 'openai':
        content = [{'type': 'input_text', 'text': user_prompt}]
        for item in images:
            content.extend([
                {'type': 'input_text', 'text': item['camera']},
                {'type': 'input_image', 'image_url': 'data:' + item['mime_type'] + ';base64,'
                 + base64.b64encode(item['data']).decode('ascii'), 'detail': config.image_detail},
            ])
        payload = {'model': config.model, 'instructions': system_prompt,
                   'input': [{'role': 'user', 'content': content}],
                   'max_output_tokens': config.max_output_tokens, 'store': False}
        if config.temperature is not None:
            payload['temperature'] = config.temperature
        return 'https://api.openai.com/v1/responses', payload

    # ============================================
    # [실험용 확장·설정 변경] Gemini는 원논문의 GPT-4o teacher와 별도 실험이다.
    # ============================================
    parts = [{'text': user_prompt}]
    for item in images:
        parts.extend([{'text': item['camera']}, {'inlineData': {
            'mimeType': item['mime_type'], 'data': base64.b64encode(item['data']).decode('ascii')}}])
    generation = {'maxOutputTokens': config.max_output_tokens}
    if config.temperature is not None:
        generation['temperature'] = config.temperature
    return ('https://generativelanguage.googleapis.com/v1beta/models/' + config.model + ':generateContent',
            {'systemInstruction': {'parts': [{'text': system_prompt}]},
             'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': generation})


# ============================================
# [실험용 확장·설정 변경] REST 통신, 오류 처리, visible-text 추출 및 제한된 재시도.
# ============================================
def http_transport(url, payload, headers, timeout):
    """stdlib REST client. API 키나 서버 오류 본문은 로그에 남기지 않는다."""
    req = request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = response.read()
    except error.HTTPError as exc:
        raise ProviderError(f'API returned HTTP {exc.code}', status=exc.code,
                            retryable=exc.code in (408, 409, 429) or 500 <= exc.code < 600) from None
    except (error.URLError, TimeoutError, ConnectionError, OSError):
        raise ProviderError('API connection failed or timed out; remote processing may have occurred',
                            retryable=True) from None
    try:
        result = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise ProviderError('API returned invalid JSON') from None
    if not isinstance(result, dict):
        raise ProviderError('API response must be a JSON object')
    return result


def response_result(config, response):
    """visible text만 반환하며 Gemini thought 및 OpenAI 내부 reasoning은 사용하지 않는다."""
    if config.provider == 'openai':
        messages = [item for item in response.get('output', []) if item.get('type') == 'message']
        parts = [part for item in messages for part in item.get('content', [])]
        return {'text': ''.join(part.get('text', '') for part in parts if part.get('type') == 'output_text'),
                'finish_reason': response.get('status', 'unknown'),
                'complete': response.get('status') == 'completed' and not any(
                    part.get('type') == 'refusal' for part in parts),
                'response_id': response.get('id'), 'returned_model': response.get('model'),
                'usage': response.get('usage', {})}
    candidates = response.get('candidates', [])
    candidate = candidates[0] if len(candidates) == 1 else {}
    parts = candidate.get('content', {}).get('parts', [])
    return {'text': ''.join(part.get('text', '') for part in parts if not part.get('thought', False)),
            'finish_reason': candidate.get('finishReason', 'blocked_or_missing'),
            'complete': candidate.get('finishReason') == 'STOP',
            'response_id': response.get('responseId'), 'returned_model': response.get('modelVersion'),
            'usage': response.get('usageMetadata', {})}


def generate(config, system_prompt, user_prompt, images, *, transport=http_transport, sleep=time.sleep):
    url, payload = build_request(config, system_prompt, user_prompt, images)
    if len(json.dumps(payload).encode('utf-8')) > config.max_request_bytes:
        raise ProviderError('Image request exceeds max_request_bytes; no API call was made')
    key = api_key(config)
    headers = {'Content-Type': 'application/json'}
    if config.provider == 'openai':
        headers['Authorization'] = 'Bearer ' + key
    else:
        headers['x-goog-api-key'] = key
    for attempt in range(1, config.max_attempts + 1):
        try:
            result = response_result(config, transport(url, payload, headers, config.timeout_seconds))
            result['attempts'] = attempt
            return result
        except ProviderError as exc:
            exc.attempts = attempt
            if not exc.retryable or attempt == config.max_attempts:
                raise
            # 전송 후 timeout은 이미 과금됐을 수 있어 exactly-once 실행은 보장하지 않는다.
            sleep(min(60.0, config.retry_base_seconds * 2 ** (attempt - 1)))
