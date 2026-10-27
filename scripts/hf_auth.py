"""Validate the process token without persisting or printing credentials."""
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    token = os.environ.get('HF_TOKEN', '')
    if not token or any(c.isspace() for c in token):
        print('올바른 HF 토큰을 입력하세요.', file=sys.stderr)
        return 1
    request = Request('https://huggingface.co/api/whoami-v2',
                      headers={'Authorization': f'Bearer {token}'})
    try:
        with urlopen(request, timeout=20) as response:
            account = json.load(response)
        if not account.get('name'):
            raise ValueError('Missing account name')
    except HTTPError as error:
        print(f'HF 인증 실패 (HTTP {error.code}). 토큰의 유효성과 권한을 확인하세요.', file=sys.stderr)
        return 1
    except (URLError, OSError, ValueError):
        print('HF 인증 서버 확인 실패. 네트워크 연결을 확인하고 다시 실행하세요.', file=sys.stderr)
        return 1
    print('HF 토큰 인증 완료. 데이터셋 이용 동의·접근 승인은 별도로 필요합니다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
