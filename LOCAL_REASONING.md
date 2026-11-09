# 로컬 추론 데이터 생성과 교체 실험

## 현재 학습 실행: config.yaml 하나로 Reason만 학습

```bash
.venv/bin/python scripts/vlm-finetuning.py --config config.yaml
```

루트 `config.yaml`의 `common`에서 batch/누적/epoch/loss 설정을, `run`에서 학습 모드·입력·split·추론 파일·출력 경로를 수정한다. 현재 기본값은 Reason만, batch 12 × 누적 1, 최대 3 epochs, 성공한 4161개 train 문항, conflict 57개 제외, validation 534/test 528 유지다. 선택 파일은 이미 생성되어 있다. 새로 설치한 환경에서 원본 생성 결과로 선택 파일을 만들어야 할 경우 `scripts/select_successful_reasoning.py`를 먼저 실행한다.

YAML의 상대 경로는 YAML 파일 위치를 기준으로 해석한다. `exact_output_dir: false`이므로 실행마다 별도 출력 폴더를 만든다. 명시한 CLI 옵션은 YAML보다 우선하며, 기존 JSON 설정도 지원한다. PyYAML은 CUDA 의존성 목록에 포함했다. 아래 Simple/Reason 동시 실행 예시는 비교가 필요할 때만 사용한다.


## 한 번에 생성

프로젝트 루트에서:

```bash
bash scripts/run_local_reasoning.sh
```

기본 모델은 `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf`, 비전 파일은 같은 저장소의 `mmproj-BF16.gguf`다. IQ3는 Q4와 다른 양자화 조건이다. 게시글의 텍스트 생성 속도를 6개 카메라 이미지 처리 속도로 해석하지 않는다.

첫 실행은 모델과 비전 파일을 다운로드하고 공식 llama.cpp CUDA 실행 파일을 준비한다. 해당 릴리스에 Linux CUDA 바이너리가 없으면 설치된 CUDA toolkit과 CMake로 소스 빌드한다. 이 PC의 `/usr/local/cuda/bin/nvcc`도 자동 탐색한다. sudo나 전역 패키지 설치는 하지 않는다. 캐시는 `.cache/local_reasoning`, 실행 파일은 `.tools/local_reasoning`에 저장한다. 외부로 보내는 것은 배포 파일 다운로드 요청뿐이며 이미지/질문 생성 요청은 localhost 서버로 보낸다.

기본 출력은 `outputs/local_reasoning/qwen38_27b_iq3/`다. 모델 커밋, llama.cpp 버전/해시, 데이터·이미지·설정 해시를 기록한다. 중단되면 **동일 명령**으로 이어서 생성한다. 이미 성공한 문항은 다시 생성하지 않는다. 설정이나 소스 코드가 바뀌면 새 `--output` 디렉터리를 사용해야 한다.

기본 동작:

1. CUDA 자동 메모리 배치, 여유 VRAM 1536 MiB. 실제 GPU 적재량은 로그로 확인한다.
2. context 32768, 동시 요청 1. 게시글의 65K 설정은 쓰지 않는다.
3. Flash Attention, Q8 KV 캐시.
4. train 8문항으로 CPU threads 4/6/8/12를 비교한다. F16 KV와 이미지 예산 512도 기록하지만 자동 선택은 원래 Q8/1024 조건 내에서만 한다. 형식 통과율은 의미 품질 검증이 아니다.
5. 동일 장면/프레임 문항을 연속 배치하고 이미지 접두부 캐시를 요청한다. 멀티모달에서 지원되지 않는 `cache-reuse`는 사용하지 않는다.
6. 이미지당 토큰 상한 1024, 출력 상한 24576(내부 추론+최종 출력 합계), 내부 reasoning 예산 -1(별도 제한 없음). 최종 설명은 최대 180단어로 제한한다. Thinking을 서버와 요청에서 명시적으로 활성화한다. 짧은 설명 제한은 최종 JSON에만 적용하며 내부 추론을 짧게 제한하지 않는다. 전체 컨텍스트/출력 상한은 여전히 존재한다. 원본 6개 이미지를 전달한다. 예산 초과나 잘린 출력은 실패로 기록한다.

처음에는 작은 표본으로 직접 확인할 수 있다:

```bash
bash scripts/run_local_reasoning.sh --skip-tuning --limit 40 --output outputs/local_reasoning/iq3_pilot
```

40문항은 **완성된 학습 데이터가 아니다**. 같은 명령에서 `--limit 40`만 제거하면 나머지를 생성한다. `--skip-tuning` 사용 여부는 실행 도중 바꿀 수 없다. 기본 실행의 최초 속도 비교는 6개 설정을 차례로 실행하므로 시간이 걸린다.

## 무엇을 생성하나

동결된 `outputs/ablation_study/20261102_controlled/split_manifest.json`의 **train 4218문항만** 생성한다. 정답은 원래 라벨을 유지하며, 모델에 이미지·질문·라벨을 주고 짧은 근거 설명을 요청한다. 내부 reasoning 출력 전체를 학습하지 않고 최종 JSON의 `explanation`만 `<think>`로 감싼다. validation/test의 추론과 정답은 원본을 유지한다.

- `events.jsonl`: 문항 ID, 생성 원문, 성공/실패, 시간. 매 문항 저장하며 재개 기준이다.
- `reasoning_overlay.json`: 기존 학습에 연결할 파일. 10문항마다 및 정상 종료 시 갱신한다.
- `status.json`: 완료/실패/남은 문항 수.
- `tuning.json`: 설정별 속도와 생성 표본. `generation.log`: 실제 CUDA 적재/서버 로그.

원본과 데이터 해시·split·문항 ID가 모두 맞고 train 전체가 성공해야 학습에 사용한다. 원본으로 조용히 대체하지 않는다. 모델이 이미지-라벨 충돌을 보고하거나 `reasoning_content`가 비어 있어 실제 Thinking 출력을 확인할 수 없으면 실패로 남긴다. 실패 기록을 검토한 후 `--retry-failed`로 재요청할 수 있지만 같은 설정에서 같은 오류가 반복될 수 있다. 형식 검사와 모델의 자기 보고만으로 시각적 정확성을 보장할 수 없으므로 표본을 사람이 검토해야 한다.

## 학습 데이터 선택

기존 보고서 실행기에 `REASONING_SOURCE` 환경변수를 추가했다. 기본값 `original`은 원래 데이터다. 새 생성 파일은 `modified` 프로필에서만 사용한다. 기존 결과 디렉터리는 덮어쓰지 않는다. 선택한 파일을 새 실행 폴더에 복사해 보존한다.

새 추론으로 Simple/Reason 학습·평가 (Instruct 제외):

```bash
EXPERIMENT_CONFIG=configs/experiment-local-reasoning-comparison.json \
REASONING_SOURCE=outputs/local_reasoning/qwen38_27b_iq3/reasoning_overlay.json \
bash scripts/run_report_experiment.sh modified
```

같은 설정으로 원본 추론 비교:

```bash
EXPERIMENT_CONFIG=configs/experiment-local-reasoning-comparison.json \
REASONING_SOURCE=original \
bash scripts/run_report_experiment.sh modified
```

위 비교 설정은 `experiment_profile=ablation`으로 명시하며 batch 12 × accumulation 1, 최대 3 epochs, 응답만 CE 학습이다. Simple은 추론을 사용하지 않는다. Reason의 train 추론만 바뀌며 질문·정답·이미지·split·평가 입력은 동일하다. validation은 원본 추론을 사용하므로 이를 실험 조건에 기록한다. 이전 full-sequence loss 실험과 단순 비교하지 말고 동일 설정의 원본 대조군을 사용한다. 추론 데이터 출처가 바뀐 체크포인트 이어 학습은 거절하며 새 학습을 시작한다.

단독 학습 스크립트에서는 `--reasoning_source original` 또는 `--reasoning_source /absolute/path/reasoning_overlay.json`을 사용할 수 있다.

## 확인

```bash
bash scripts/run_local_reasoning.sh --dry-run
.venv/bin/python -m unittest discover -s tests
```

전자는 원본 데이터/split/이미지를 확인하며 다운로드나 GPU 실행을 하지 않는다. 단위 테스트는 연결·재개·출력 형식 계약을 검증하며 모델 속도/정확도 검증과는 별개다.

## 실제 실행 확인 (2026-11-07)

RTX 5070 Ti에서 IQ3 + BF16 비전 파일, Thinking ON, 추론 예산 -1, 32K context로 6개 이미지가 있는 train 문항 3803을 생성했다. 요청 약 11.95초, prompt 6476토큰, 출력 325토큰, 서버의 출력 생성 구간 약 55.23토큰/초였다. `reasoning_content` 존재, 정상 종료와 최종 JSON 형식을 확인했다. 이는 한 문항 실행 확인이며 의미 정확성이나 전체 평균 속도를 검증한 결과가 아니다. 전체 4218문항 생성 및 새 데이터 학습은 아직 실행하지 않았다. 세부 기록: `outputs/local_reasoning/iq3_smoke/VERIFICATION.json`.

## 속도 표본의 conflict 처리 수정

속도 비교는 정상 종료·Thinking 존재·최종 JSON 형식을 확인한다. `evidence_status=conflict`는 정상적인 검토 결과이므로 속도 후보를 탈락시키지 않는다. 반면 학습용 생성에서는 여전히 실패/검토 대상으로 남기며 학습에 자동 포함하지 않는다. 모델의 conflict 표시는 원본 라벨 오류의 확정 판정이 아니다.

이미 속도 비교만 마친 상태에서 이 수정으로 코드 해시가 바뀐 경우 `bash scripts/run_local_reasoning.sh --repair-tuning-only`로 기존 측정을 복구할 수 있다. 데이터·이미지·모델·프롬프트·설정·보조 코드가 동일하고, 본 생성 기록과 설정 선택 결과가 없는 경우에만 허용한다. 이전 manifest와 측정 원문은 백업한다. 복구 후에는 원래 명령으로 실행한다. 이미 본 생성을 시작한 실행의 코드 변경을 허용하는 기능은 아니다.

## 충돌 문항 제외 학습 (사용자 선택)

`select_successful_reasoning.py`는 원본 생성 파일을 보존하고 `reasoning_success_only.json`을 만든다. 2026-11-09 실행에서는 성공 4161문항을 선택하고 conflict 57문항을 제외했다. 미처리 문항이나 다른 생성 오류를 임의로 제외하지 않는다. 제외 ID와 원본 파일 SHA256을 기록하며, 모든 train ID가 선택/제외 중 하나로 설명되어야 한다.

```bash
.venv/bin/python scripts/select_successful_reasoning.py
EXPERIMENT_CONFIG=configs/experiment-local-reasoning-comparison.json \
REASONING_SOURCE=outputs/local_reasoning/qwen38_27b_iq3/reasoning_success_only.json \
REPORT_RESULTS_ROOT=outputs/local_reasoning_experiments \
bash scripts/run_report_experiment.sh modified
```

Simple과 Reason 모두 동일한 4161개 train 문항으로 학습한다. validation/test는 원래 split 그대로 유지하며 train 제외 목록은 각 모델의 training_config.json과 skipped_samples.json에 남긴다. 전체 train을 사용한 이전 실험과는 데이터 개수가 다르므로 구분해서 해석한다. 생성 코드/학습 보조 코드의 변경으로 기존 생성 실행의 코드 해시도 달라질 수 있으므로, 추가 생성이 필요하면 새 출력 경로를 사용한다.

Git에는 코드·설정·테스트·문서를 보관하고, outputs 및 모델·데이터·환경·캐시·비밀 설정은 제외한다. 이미 추적되던 outputs는 로컬 파일을 남기고 인덱스에서 제외했다. 기존 커밋 기록은 재작성하지 않았다.
