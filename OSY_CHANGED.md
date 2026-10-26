# ReasonDrive C1 변경 내역

기준일: 2026-10-26. 원본 비교 기준은 [저자 공개 저장소](https://github.com/zhourobotics/reasondrive)의 `326caecafedad1b9dbc156ebd77c8dc0eb998740`, 이번 작업 전 로컬 기준은 `e38288ab3c59c02c1b28377ef27067c2b8104db6`이다. 아래는 **코드 구현 상태**이며 새 학습의 성능 재현 결과가 아니다. 이번 작업에서는 README를 수정하지 않았다.

## 목표와 주석 분류

C1은 학습·평가 데이터가 겹치지 않는 조건에서 Instruct, Simple, Reason을 비교할 수 있도록 재현 절차를 구성하는 작업이다. C2의 이미지 개입과 C3의 reasoning 개입 실험은 이번 범위가 아니다. Simple과 Reason은 각각 같은 기반 모델에서 새로 학습한다. Instruct는 추가 학습 없이 기반 모델을 평가한다.

새 파일은 맨 위에, 부분 변경은 해당 위치에 다음 구분을 남긴다. 공개 코드에 없다는 이유만으로 모든 보조 도구를 논문 재구현으로 부르지는 않는다.

```python
# ============================================
# 논문 기반 재구현 — 파일 전체 또는 해당 부분
# 논문에는 있지만 공개 코드에 없던 절차
# ============================================

# ============================================
# 공개 코드 오류 수정 — 해당 부분
# 원본 공개 코드에서 확인한 문제와 수정 이유
# ============================================

# ============================================
# 실험용 확장·설정 변경 — 파일 전체 또는 해당 부분
# 실험을 위해 우리가 추가하거나 선택한 조건
# ============================================
```

로컬 평가기·추출기의 결함은 **재구현부 보완**이라고 덧붙인다. 이를 원저자 공개 코드의 오류로 표시하지 않는다. GPT 채점은 미구현 상태를 유지하며 OpenAI/Gemini 연결은 reasoning 생성기에만 있다.

## 이전부터 반영되어 있던 수정

| 구분 | 원본 공개 코드와 현재 구현의 차이 |
| --- | --- |
| 공개 코드 오류 수정 | 원본은 학습 seed 3407과 추론 seed 3047로 QA를 각각 나눴다. 현재는 장면 단위 train/validation/test 분할을 저장하고 재사용한다. 데이터 hash·행 번호를 검증하며 test를 모델 선택에 사용하지 않는다. |
| 공개 코드 오류 수정 | 입력 변환 실패로 생긴 None을 학습 데이터에 남기지 않는다. collator 4096과 학습 설정 2048의 불일치를 하나의 설정으로 맞췄다. |
| 공개 코드 오류 수정 | 학습 스크립트의 예시 추론에서 prompt 토큰을 제외한다. **원본 최종 추론 코드도 이미 prompt 토큰과 GT 입력을 제외했다.** 이 부분까지 신규 수정으로 주장하지 않는다. |
| 논문 기반 재구현 | 원본에 있던 Reason 학습에 Simple 비교 모드를 추가했다. 누락된 평가 코드를 DriveLM 기준에 따라 재구현했다. |
| 실험용 확장 | Mac MLX 실행, 자동 다운로드·검증, 장면 분할, 사전 입력 검사, 실패 기록, 실행 스크립트를 추가했다. |

원본에도 224×224 이미지 처리, QLoRA, 언어 MLP 학습, Reason 학습과 주요 하이퍼파라미터가 있었다. 이를 전부 새로 추가했다고 표현하지 않는다. 공개 코드의 문제를 확인한 것이며 논문 저자의 실제 실험 오류를 입증한 것은 아니다.

thinking 형식 때문에 제외되던 train 19개·validation 5개는 **이전 로컬 전처리기의 형식 검사**에서 발생한 제외였다. 태그 밖 설명·반복 태그·닫는 태그 누락·`<thinkThe` 오타를 원문 문장과 좌표를 보존하며 정규화한다. 설명의 사실성이나 거절 문구를 고친 것은 아니다. 이 복구와 모델 생성 답변 검사는 서로 다르다.

## 이번에 구현한 수정

### 누락된 그래프 시작 질문 복원

현재 test 80프레임 중 5프레임에는 DriveLM 객체 목록 시작 질문이 없으며 해당 프레임에 원래 test 28문항이 속한다. 이전 로컬 평가기는 첫 perception 질문을 시작 질문으로 가정해 일부 프레임에서 중단되거나 잘못된 그래프를 만들었다.

- `scripts/graph_context.py`는 원본 DriveLM 주석에서 scene/frame별 시작 질문을 찾고 이미지 일치·파일 SHA-256을 검증한다.
- `prepare_inputs.py`가 학습 전에 graph context를 준비한다. 문맥이 없으면 학습·추론 전에 오류로 안내한다.
- CUDA/MLX 추론기는 누락된 시작 질문을 별도로 추론한다. 모델 입력은 이미지·질문뿐이며 정답 객체 목록은 넣지 않는다.
- 별도 응답은 `.context.json`에 저장한다. 원래 test **528문항은 유지**하고 추가 5개 질문은 점수에 넣지 않는다. 해당 프레임을 몰래 제외하거나 정답 그래프로 예측 그래프를 대체하지 않는다.
- 원본 `v1_1_train_nus.json`은 192,961,829바이트(약 193 MB)이며 SHA-256은 `ecd3c13e9beb27295bcf6182ea35fff7c1d6a9ba58d40e48fb38d713094759fd`이다. 최초 준비 시 이 주석 다운로드가 추가된다.

### 생성 답변의 엄격한 추출

`data_utils.py`의 생성 응답 전용 parser는 비어 있지 않은 `<answer>...</answer>` 쌍을 정확히 하나 요구한다. 중복·중첩·불완전한 answer 태그와 think 내부의 answer를 거부한다. 기존 reasoning 정규화로 생성 결과의 형식 오류를 감추지 않는다. 원본 응답·실패 상태·오류를 보존하고 채점 때도 원본 응답을 다시 검사한다.

### Instruct 기준선과 학습 조건

`configs/experiment.json`과 `scripts/experiment_config.py`를 공통 조건의 원천으로 사용한다. 두 실행 스크립트에 Instruct 추론을 추가했다. Instruct와 Simple은 답변만, Reason은 reasoning과 답변을 요구한다. 세 모델의 split·이미지 조건·greedy 생성·최대 생성 길이는 같다.

| 공통 조건 | 기본값 |
| --- | --- |
| 기반 모델 | Qwen2.5-VL-3B-Instruct 계열 4bit, backend별 변환본과 commit revision 고정 |
| epoch / batch / 누적 | 3 / 1 / 12 |
| 학습률 / weight decay | 2e-4 / 0.01 |
| LoRA | 언어 MLP gate/up/down projection, rank 8, alpha 8, dropout 0 |
| split seed | 3407 |
| 이미지 / 최대 학습 길이 | 6장 각각 224×224 / 2048토큰 |
| 생성 | greedy, 최대 512토큰 |
| loss 정책 | CUDA `train_on_responses_only=False`, MLX `train_on_completions=False` |
| 모델 선택 | epoch별 validation, 최저 validation loss, 2회 미개선 early stopping |

False는 질문·지시문 텍스트도 loss 대상에 포함한다. CUDA collator는 image·padding 토큰을 별도로 마스킹하지만, MLX-VLM 0.7.8의 full-sequence loss에는 attention mask가 1인 이미지 placeholder도 포함된다. 이 차이는 vision_loss_policy로 기록하며 양쪽 loss가 완전히 같다고 주장하지 않는다. 이는 논문에 명시된 마스킹 규칙이라고 단정한 것이 아니라 공개 CUDA 구현의 방향과 이번 사용자 선택을 따른 것이다. 이전 MLX의 completion-only 설정과 기존 adapter는 새 실험 조건과 구분해야 한다. 기반 비전·attention을 학습한다는 뜻은 아니다.

CUDA 단독 학습 기본값도 3B로 맞췄다. 모델과 processor는 같은 고정 snapshot에서 읽고 실제 config·hash·revision·패키지·backend 정보를 저장한다. 모델 ID 변경 시 revision도 지정해야 한다.

CUDA는 bitsandbytes 4bit, paged AdamW 8bit, 선형 스케줄·warmup 50을 사용한다. MLX는 MLX 4bit, AdamW, 고정 학습률·warmup 없음이다. 서로 완전히 동일한 backend 실험이나 논문 수치의 정확한 재현이라고 표시하지 않는다. 장면 분할도 C1을 위한 실험 설계 변경이다.

### OpenAI와 Gemini reasoning 생성기

`generate_reasoning.py`와 `reasoning_providers.py`는 논문 Table 1의 perception·prediction·planning·behavior별 프롬프트와 이미지 6장·질문·정답으로 reasoning을 생성한다. 기존 정답과 원본 metadata를 보존한다.

논문 GPT-4o 절차는 **논문 기반 재구현**, provider·모델 선택과 재시도·재개·운영 기록은 **실험용 확장**이다. Gemini 생성 데이터는 논문의 GPT-4o 조건과 동일하게 간주하지 않는다. 키는 환경변수로 받고 결과에 저장하지 않는다. 기본 dry-run은 API를 호출하지 않으며 실제 생성은 `--execute`를 명시한다. 실패·미완료 행은 성공 데이터로 위장하지 않는다.

```bash
# API 호출 없는 입력 검사
python scripts/generate_reasoning.py --data_path data/drivelm.json --image_root data/nuscenes --output_path data/reasoning_openai.json --config configs/reasoning_openai.example.json --dry_run

# OPENAI_API_KEY가 설정된 환경에서만 실제 생성
python scripts/generate_reasoning.py --data_path data/drivelm.json --image_root data/nuscenes --output_path data/reasoning_openai.json --config configs/reasoning_openai.example.json --execute --limit 10

# 같은 출력·설정의 미완료 작업 재개
python scripts/generate_reasoning.py --data_path data/drivelm.json --image_root data/nuscenes --output_path data/reasoning_openai.json --config configs/reasoning_openai.example.json --execute --resume --retry_failed
```

Gemini는 `configs/reasoning_gemini.example.json`과 `GEMINI_API_KEY`를 사용한다. 입력 파일 덮어쓰기는 거부한다. 재개 시 데이터·이미지·프롬프트·모델 조건이 바뀌면 오류로 안내한다. 기본 자동 학습 파이프라인이 유료 API를 호출하지는 않는다. 재생성 데이터를 학습할 때는 해당 파일과 새 hash에 맞는 분할·출력 경로를 지정해야 한다.

## 평가 지표와 해석

공개 저장소가 안내하던 `scripts/evaluation.py`는 없었다. 현재 평가기는 공식 DriveLM QA tag·좌표 그래프를 참고한 재구현이며 저자의 비공개 코드와 완전히 같다고 보장하지 않는다.

- Accuracy: tag 0 문자열 일치.
- BLEU-1~4, ROUGE-L, CIDEr: tag 2, 공식 COCO backend.
- Match_noGPT: tag 3 좌표 매칭 F1.
- GPT 전용 tag 1 제외. reasoning 자체는 채점하지 않는다.

```text
Language = (mean(BLEU-1~4) + ROUGE-L + CIDEr/10) / 3
Match_noGPT = 좌표 매칭 F1
Final_noGPT = (Accuracy + Language + Match_noGPT) / 3
```

Match_noGPT/Final_noGPT는 논문의 GPT 포함 Match/Final Score와 직접 비교할 수 없다. 예측 객체 그래프에 따라 실제 채점 수가 모델마다 달라질 수 있으므로 counts·그래프 제외 수를 함께 확인한다. 실패 행을 삭제해 비교하지 않는다. 해당 지표의 대상이 없으면 N/A다.

DriveLM 그래프 기준 commit은 `1de72a74b257e5373400fa68239e99bd5d20580a`, language-evaluation은 `8e9e03feb7f5e605f20fc0c8dba0e24532a26216`이다. vendor 알고리즘과 라이선스를 유지했다. PTB tokenizer에는 Java가 필요하다.

## 실행과 재현성

Mac은 프로젝트 루트에서 `bash scripts/run_mac.sh`, 네이티브 Ubuntu 24.04 x86_64 + NVIDIA는 `bash scripts/run_ubuntu.sh`를 실행한다. 기존 이미지 경로를 첫 인자로 줄 수 있다. **이번에는 전체 실행을 수행하지 않았다.**

향후 실행 순서는 환경 준비 → 데이터 준비 → 입력·그래프 문맥 검사 → 시험 학습 → Simple/Reason 학습 → Instruct/Simple/Reason 추론 → 비교표 생성이다. 시험 adapter는 최종 평가에서 거부한다. 기존 출력은 보존하고 새 폴더를 사용한다. 시험 학습 성공이 전체 학습·최장 샘플·성능 재현의 성공을 뜻하지는 않는다.

- `EPOCHS=1`, Mac의 `GRAD_CHECKPOINT=0`은 학습량·메모리·시간에 영향을 주므로 별도 조건으로 기록한다.
- 설치 결과는 `requirements-resolved.txt`에 저장한다. CUDA requirements는 호환성 범위이며 **실기 검증된 고정 lockfile이 아니다**. 동결 버전 재설치도 GPU·driver·CUDA 환경의 동등성을 보장하지 않는다.
- train/validation에서 길이 제한으로 제외된 행과 캐시는 실제 실행 기록에서 확인한다. 고정 split의 4,218/534문항은 optimizer가 실제 본 행 수와 구분한다.
- 데이터·이미지 revision을 고정하고 출처를 기록한다. 이미지 archive와 모델에 수 GB 공간이 추가로 필요하다.

공통 설정 파일은 `EXPERIMENT_CONFIG=/path/to/config.json`, 저장한 패키지 목록 재설치는 `DEPENDENCY_SNAPSHOT=/path/to/requirements-resolved.txt` 환경변수로 지정한다. 예를 들어 `DEPENDENCY_SNAPSHOT=/path/to/requirements-resolved.txt bash scripts/run_mac.sh`로 기존 버전 목록을 설치 입력으로 사용할 수 있다. 모델을 바꾸려면 `MODEL_ID`와 `MODEL_REVISION`을 함께 지정한다.

## 검증 상태

- 단위·회귀 테스트 **50개 통과**. 생성기 mock 17개, strict parser·graph 문맥·비교 조건·Instruct 무학습 분기·기존 입력 처리 등을 포함한다.
- Python 파일 20개 구문 검사, 실행 셸 2개 `bash -n`, `git diff --check` 통과. README는 작업 전 HEAD와 내용이 같다.
- 실제 ReasonDrive test 528문항과 복원 context 5개에 대해 정답을 예측으로 넣은 **오프라인 평가 경로 검사** 통과. dataset hash와 test 행 번호는 유지했고 그래프 제외 0개였다. 이 검사의 언어 backend는 stub이며 모델 추론이나 언어 점수 검증 결과가 아니다.
- 위 검사에서 tag별 대상은 Accuracy 205개, language 75개, coordinate match 27개, GPT-only 제외 221개였다. 실제 모델 응답의 대상 수는 그래프 필터 결과에 따라 달라진다.
- API 키 없는 CLI dry-run에서 이미지 6개의 입력을 확인하고 API 호출 0회·생성 데이터 미작성 확인. mock 테스트 외 실제 provider 연결은 실행하지 않았다.

개발 환경에서 테스트는 `python -m unittest discover -s tests -p 'test_*.py'`, 셸 구문은 `bash -n scripts/run_mac.sh` 및 `bash -n scripts/run_ubuntu.sh`로 검사한다. 이번 검사 요약은 `reports/validation_20261026.json`에 저장했다.

실제 CUDA·MLX 학습, 전체 test 추론, 유료 OpenAI/Gemini 호출, 성능·메모리·속도 측정은 요청에 따라 생략했다. 과거 로그의 4,199/529행, completion-only loss, 시험 학습 peak 메모리와 테스트 17/19개 통과 기록은 이번 조건의 검증 결과로 재사용하지 않는다. 기존 checkpoint의 중복 데이터·loss 조건도 소급 변경되지 않는다.

## 참고한 공개 issue

- [#1 runtime comparison](https://github.com/zhourobotics/reasondrive/issues/1): 실행 시간 비교 문의. 이번에는 속도를 측정하지 않았다.
- [#2 evaluation error](https://github.com/zhourobotics/reasondrive/issues/2): 누락된 평가기·낮은 Accuracy 관련 문의. 우리의 구현이 게시된 모든 환경의 문제를 해결했다고 단정하지 않는다.
- [#3 model loading](https://github.com/zhourobotics/reasondrive/issues/3): 공개 모델 loader 예제 문제. 이번에는 README를 수정하지 않도록 요청받아 예제는 그대로 두었다. 실행 코드는 VLM용 loader를 사용한다.

이슈의 존재는 논문 결과가 틀렸다는 증거가 아니다. 확인한 코드 차이, 선택한 조건, 미실행 검증을 구분한다.
