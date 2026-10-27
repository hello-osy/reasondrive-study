#!/usr/bin/env bash
# Source from launchers; children inherit the token without another prompt.
require_hf_token() {
  set +x
  if [[ -z ${HF_TOKEN:-} ]]; then
    if [[ ! -t 0 ]]; then
      echo 'HF 토큰이 필요합니다. 터미널에서 실행하거나 HF_TOKEN 환경변수를 설정하세요.' >&2
      return 1
    fi
    echo 'Hugging Face Read 토큰을 입력하세요 (화면에 표시되지 않음).' >&2
    echo '토큰 발급: https://huggingface.co/settings/tokens' >&2
    IFS= read -r -s -p 'HF token: ' HF_TOKEN || { echo >&2; return 1; }
    echo >&2
  fi
  if [[ -z ${HF_TOKEN:-} ]]; then
    echo '빈 토큰으로는 실행할 수 없습니다.' >&2
    return 1
  fi
  export HF_TOKEN
  export HF_HUB_DISABLE_IMPLICIT_TOKEN=0
  if [[ ${REASONDRIVE_HF_AUTH_CHECKED:-} != 1 ]]; then
    python3 "${BASH_SOURCE[0]%/*}/hf_auth.py" || return 1
    export REASONDRIVE_HF_AUTH_CHECKED=1
  fi
}
