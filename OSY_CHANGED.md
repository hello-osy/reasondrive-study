# ReasonDrive 변경 내역

## 실험 명령어(복붙)

<details>
<summary>1. Ubuntu 실행 환경 준비</summary>

모든 명령은 `reasondrive-study` 루트에서 실행한다. 비교 실험은 **Ubuntu 24.04 x86_64 + NVIDIA GPU** 전용이며, `nvidia-smi`가 정상 동작하는 드라이버 환경이 필요하다.

```bash
sudo apt update
sudo apt install -y curl git openjdk-17-jre-headless

# uv가 없다면 프로젝트 안에 설치한다.
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=.tools/uv UV_NO_MODIFY_PATH=1 sh
export PATH=".tools/uv:$PATH"

uv python install 3.11
uv --version
java -version
nvidia-smi
```

같은 터미널에서 아래 실험 명령을 실행한다. 새 터미널에서는 루트에서 `export PATH=".tools/uv:$PATH"`를 다시 실행한다. 이미 PATH에서 `uv`를 사용할 수 있다면 설치와 PATH 설정은 생략한다. 설치 옵션은 [uv 공식 문서](https://docs.astral.sh/uv/reference/installer/)를 따른다.

실행 스크립트가 Python 3.11의 `.venv` 생성, CUDA 의존성 설치·검사, 데이터·모델 준비를 처리한다. Java 17은 평가의 PTB tokenizer에 사용한다. 모델·이미지·출력 저장에 수 GB 이상의 여유 공간이 필요하다.

</details>

<details>
<summary>2. 공개 코드 조건 / 수정 조건으로 각각 새 학습</summary>

두 실험 모두 **같은 고정 3B 기반 모델에서 Simple과 Reason을 각각 새로 학습**한다. 공개 Simple/Reason 가중치를 다운로드해 평가하는 명령이 아니다.

```bash
# 실험 1: 공개 코드 조건 + Simple 최소 확장
bash scripts/run_report_experiment.sh public_baseline

# 실험 2: 현재 수정 코드 조건
bash scripts/run_report_experiment.sh modified

# 두 실험을 차례로 실행
bash scripts/run_report_experiment.sh both

# 기존 이미지 경로를 사용할 때: 루트 기준 상대경로
bash scripts/run_report_experiment.sh both data/nuscenes

# 학습·추론 없이 실행 계획 확인
bash scripts/run_report_experiment.sh both --dry-run
```

`public_baseline`은 원본의 QA 분할·학습 설정·sampling을 유지한다. `modified`는 장면 분할·validation 기반 모델 선택·greedy 생성·엄격한 답 추출을 사용한다. 상세 차이는 아래 「바뀐 점」의 비교표에 정리했다.

매 실행마다 새 디렉터리를 만들고 이전 결과를 보존한다. **중단된 학습 checkpoint에서 자동 재개하는 기능은 지원하지 않는다.** 다시 실행하면 새 학습을 시작한다. 두 실험의 기본 파이프라인은 유료 OpenAI/Gemini API를 호출하지 않는다.

</details>

<details>
<summary>3. 결과 위치와 재집계</summary>

| 결과 | 루트 기준 상대경로 |
| --- | --- |
| 각 실행의 비교표 | `outputs/report_experiments/{profile}/<run_id>/comparison/comparison.md` |
| 같은 결과의 CSV / JSON | 위 경로의 `comparison.csv` / `comparison.json` |
| profile별 최근 완료 실행 경로 | `outputs/report_experiments/{profile}/latest.json` |
| 두 실험의 합본 | `outputs/report_experiments/comparison.md` / `comparison.csv` / `comparison.json` |

`{profile}`은 `public_baseline` 또는 `modified`다. 최근 완료 경로는 `latest.json`, 각 실행의 진행·실패 상태는 해당 실행 디렉터리의 `status.json`에서 확인한다. 이전 실행 디렉터리는 삭제하지 않는다.

```bash
# 저장된 결과만 다시 모아 비교표 생성
.venv/bin/python scripts/report_experiment.py summarize --root outputs/report_experiments
```

두 profile은 test 구성과 학습 데이터 중복 조건이 다르다. 합본은 각 profile의 최근 완료 실행에 대한 점수·대상 수·조건을 나란히 정리하며 **단순 개선율을 출력하지 않는다.** 실패한 실행은 집계하지 않으며, 이전 완료 실행이 있으면 그 결과가 남는다. 두 profile 모두 완료 이력이 없으면 재집계할 수 없다. 점수만으로 코드 수정의 효과를 분리해 입증하려면 같은 평가 집합을 사용하는 별도 통제 실험이 필요하다.

</details>

<details>
<summary>4. 기존 CUDA / Mac 수정 파이프라인과 설정 변경</summary>

기존 실행 경로도 유지한다. 두 경로는 수정 조건에서 Simple/Reason을 학습하고, 추가 학습이 없는 Instruct와 함께 평가한다.

```bash
# Ubuntu CUDA 수정 파이프라인
bash scripts/run_ubuntu.sh

# Apple Silicon Mac MLX 수정 파이프라인
bash scripts/run_mac.sh

# 기존 이미지 경로 지정
bash scripts/run_mac.sh data/nuscenes
```

`run_report_experiment.sh`의 두 실험 비교는 CUDA 전용이다. Mac 결과와 CUDA 결과를 동일한 backend의 비교 실험으로 취급하지 않는다.

기존 `run_ubuntu.sh` / `run_mac.sh`는 `EXPERIMENT_CONFIG`로 공통 설정 파일, `DEPENDENCY_SNAPSHOT`으로 저장한 패키지 목록을 지정할 수 있다. 예시 경로의 파일을 준비한 뒤 실행한다.

```bash
EXPERIMENT_CONFIG=configs/experiment.json bash scripts/run_mac.sh
DEPENDENCY_SNAPSHOT=outputs_previous/requirements-resolved.txt bash scripts/run_mac.sh
```

모델을 바꾸려면 `MODEL_ID`와 `MODEL_REVISION`을 함께 지정한다. `EPOCHS=1`, Mac의 `GRAD_CHECKPOINT=0`은 학습량·메모리·시간에 영향을 주는 별도 조건이다. 기존 adapter와 출력의 분할·loss 조건은 소급해서 바뀌지 않는다.

</details>

<details>
<summary>5. reasoning 데이터 생성기: 선택적으로 API 사용</summary>

OpenAI/Gemini 연결은 **reasoning 데이터 생성에만** 사용한다. 기본 dry-run은 API를 호출하지 않으며, 실제 생성에는 키 설정과 `--execute`가 필요하다. 아래 명령은 자동 학습 파이프라인과 별개다.

```bash
# API 호출 없는 입력 검사
.venv/bin/python scripts/generate_reasoning.py --data_path data/drivelm.json --image_root data/nuscenes --output_path data/reasoning_openai.json --config configs/reasoning_openai.example.json --dry_run

# OPENAI_API_KEY가 설정된 환경에서 실제 10건 생성
.venv/bin/python scripts/generate_reasoning.py --data_path data/drivelm.json --image_root data/nuscenes --output_path data/reasoning_openai.json --config configs/reasoning_openai.example.json --execute --limit 10

# 같은 출력·설정의 미완료 생성 작업 재개
.venv/bin/python scripts/generate_reasoning.py --data_path data/drivelm.json --image_root data/nuscenes --output_path data/reasoning_openai.json --config configs/reasoning_openai.example.json --execute --resume --retry_failed
```

Gemini는 `configs/reasoning_gemini.example.json`과 `GEMINI_API_KEY`를 사용한다. 입력 파일 덮어쓰기를 거부하며, 재개 시 데이터·이미지·prompt·모델 조건이 바뀌면 오류로 안내한다. 생성 작업의 재개는 학습 checkpoint 재개와 별개다. 재생성 데이터를 학습하려면 해당 파일과 새 hash에 맞는 분할·출력 경로를 지정해야 한다.

</details>

<details>
<summary>6. 개발용 검사 명령</summary>

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
bash -n scripts/run_report_experiment.sh
bash -n scripts/run_ubuntu.sh
bash -n scripts/run_mac.sh
git diff --check
```

이 명령은 검사 방법이다. 실제 통과 이력과 미실행 범위는 아래 「검증 상태」를 확인한다.

</details>

## 바뀐 점

<details>
<summary>목적과 변경 분류</summary>

논문에 있지만 공개 코드에 없던 기능을 재구현하고, 공개 코드의 오류를 수정하며, 실험에 필요한 설정을 추가해 ReasonDrive 재현과 후속 실험의 baseline을 마련한다. C1은 겹치지 않는 학습·평가 조건에서 Instruct/Simple/Reason을 비교하는 범위이며, C2의 이미지 개입과 C3의 reasoning 개입 실험은 여기서 구현하지 않는다. 공개 코드 조건 비교는 수정 조건과 원본 조건의 차이를 기록하는 별도 profile이다.

새 파일은 맨 위, 일부 변경은 해당 위치에 다음 분류를 남긴다.

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

로컬 평가기·추출기의 결함 수정에는 **재구현부 보완**을 덧붙인다. 이를 원저자 공개 코드의 오류로 표시하지 않으며, 모든 보조 도구를 논문 기반 재구현으로 분류하지 않는다. GPT 채점은 미구현 상태를 유지한다. README는 이번 수정 대상에서 제외했다.

</details>

<details>
<summary>추가한 두 실험 profile의 차이</summary>

`scripts/run_report_experiment.sh`와 `scripts/report_experiment.py`가 환경 준비·새 학습·추론·채점·결과 집계를 연결한다. `scripts/public_baseline.py`는 보존한 공개 원본을 실행하는 연결부다.

| 항목 | `public_baseline` | `modified` |
| --- | --- | --- |
| 실행 코드 | 원본 commit `326caec` + Simple 최소 확장 + 라이브러리 인자명 호환 | 현재 수정 코드 |
| Simple / Reason | 같은 기반 모델에서 각각 새 학습 | 같은 기반 모델에서 각각 새 학습 |
| 기반 모델 | 두 profile이 같은 고정 Qwen2.5-VL-3B snapshot 사용 | 왼쪽과 동일 |
| 데이터 분할 | 원본 QA 단위 90/10, 학습 seed 3407·추론 seed 3047 | 장면 단위 80/10/10, 저장한 split 재사용 |
| validation / 저장 | 원본 `eval_steps=10`, `save_steps=50` 유지 | epoch별 validation loss로 모델 선택 |
| 길이 설정 | 원본 collator 4096 / 학습 2048 불일치 유지 | 공통 2048 설정 |
| 생성 | 원본 sampling, temperature 0.7 / top-p 0.9 | greedy |
| 답변 처리 | 원본 생성 응답 보존, 공통 채점 단계에서 엄격한 답 추출 | 추론 시 형식 상태 기록, 공통 채점 때 재검사 |
| 평가 | 공통 재구현 noGPT 평가기 | 공통 재구현 noGPT 평가기 |

원본은 학습·추론에서 QA를 따로 나누므로 학습 데이터와 평가 데이터가 겹칠 수 있다. `public_baseline`은 이 조건과 원본의 입력 실패 처리 등 확인된 문제를 그대로 남긴다. 라이브러리 인자명 호환을 추가했다고 해서 원본의 모든 실행 오류를 해결한 것은 아니다.

Simple은 공개 원본에 없으므로 학습 정답과 지시문에서 reasoning을 제거하는 최소 확장으로 실행한다. 실제 적용한 호환 변경과 실행 조건은 기록한다. 원본 그대로의 무수정 실행 또는 저자의 비공개 실험 재현이라고 부르지 않는다.

공통 noGPT 평가기, 누락된 그래프 시작 질문의 별도 추론, 원본 응답·실패 행 보존은 비교 결과를 수집하기 위해 추가한 절차다. 별도로 추론한 시작 질문은 채점 대상에서 제외한다. 원본과 수정본을 같은 평가 구현에 연결하더라도 두 profile의 분할·생성 조건 차이는 남는다.

</details>

<details>
<summary>이전부터 반영된 수정과 원본에 이미 있던 기능</summary>

| 분류 | 수정 조건에 반영한 내용 |
| --- | --- |
| 공개 코드 오류 수정 | 학습 seed 3407·추론 seed 3047의 독립 QA 분할을 장면별 train/validation/test manifest로 대체했다. 데이터 hash·행 번호를 검증하고 test는 모델 선택에 쓰지 않는다. |
| 공개 코드 오류 수정 | 변환 실패로 생긴 `None`을 학습 데이터에서 제외하고, collator 4096 / 학습 2048 불일치를 공통 설정으로 맞췄다. |
| 공개 코드 오류 수정 | 학습 스크립트의 예시 추론에서 prompt 토큰을 제외했다. 원본 최종 추론 코드도 이미 prompt 토큰과 GT 입력을 제외했으므로 이를 신규 수정으로 주장하지 않는다. |
| 논문 기반 재구현 | 원본 Reason 학습에 Simple 비교 모드를 추가하고, 누락된 평가기를 DriveLM 기준으로 재구현했다. |
| 실험용 확장·설정 변경 | Mac MLX 실행, 자동 다운로드·검증, 장면 분할, 입력 검사, 실패 기록과 실행 자동화를 추가했다. |

원본에도 224×224 이미지 처리, QLoRA, 언어 MLP 학습, Reason 학습과 주요 하이퍼파라미터가 있었다. 장면 분할은 중복 문제를 막는 수정이면서 C1의 실험 설계 변경이다. 공개 코드의 문제를 확인한 것이며 저자의 실제 논문 실험에 오류가 있었다는 증거는 아니다.

thinking 형식 때문에 제외되던 train 19개·validation 5개는 **이전 로컬 전처리기의 검사**에서 제외된 행이다. 태그 밖 설명·반복 태그·닫는 태그 누락·`<thinkThe` 오타를 원문 문장과 좌표를 보존하며 정규화했다. 설명의 사실성이나 거절 문구를 고친 것은 아니며, 아래 생성 응답의 엄격한 검사와 구분한다.

</details>

<details>
<summary>재구현부 보완: 그래프 시작 질문 복원과 엄격한 답 추출</summary>

수정 조건의 test 80프레임 중 5프레임에는 DriveLM 객체 목록 시작 질문이 없고, 해당 프레임에는 원래 test 28문항이 속한다. 이전 로컬 평가기는 첫 perception 질문을 시작 질문으로 가정해 중단되거나 잘못된 그래프를 만들었다.

- `scripts/graph_context.py`가 원본 DriveLM 주석에서 scene/frame별 시작 질문을 찾고 이미지 일치와 파일 SHA-256을 검증한다.
- `scripts/prepare_inputs.py`가 학습 전에 graph context를 준비한다. 문맥이 없으면 학습·추론 전에 오류로 안내한다.
- CUDA/MLX 추론기는 누락된 시작 질문에 이미지·질문만 넣고 별도로 응답을 생성한다. 정답 객체 목록은 입력하지 않는다.
- 응답은 `.context.json`에 별도 저장한다. 원래 test **528문항을 유지**하고 추가 5개 문맥 질문은 점수에서 제외한다. 해당 프레임을 삭제하거나 예측 그래프를 정답 그래프로 대체하지 않는다.
- 원본 `v1_1_train_nus.json`은 192,961,829바이트(약 193 MB), SHA-256은 `ecd3c13e9beb27295bcf6182ea35fff7c1d6a9ba58d40e48fb38d713094759fd`다. 최초 준비 시 이 주석 다운로드가 추가된다.

`scripts/data_utils.py`의 생성 응답 parser는 비어 있지 않은 `<answer>...</answer>` 쌍을 정확히 하나 요구한다. 중복·중첩·불완전한 answer 태그와 think 내부의 answer를 거부한다. 데이터 정규화로 생성 결과의 형식 오류를 감추지 않으며, 원본 응답·실패 상태·오류를 보존하고 채점 때 원본을 다시 검사한다. 두 실험의 채점에 같은 엄격한 응답 검사를 적용한다. 공개 코드 기반 실험은 원본 생성 응답을 변경하지 않고 수집·채점 단계에서 검사한다.

</details>

<details>
<summary>Instruct baseline과 수정 조건의 학습 설정</summary>

`configs/experiment.json`과 `scripts/experiment_config.py`가 수정 조건의 공통 설정 원천이다. 기존 CUDA/MLX 파이프라인에 추가한 Instruct는 도메인 추가 학습 없이 기반 모델을 평가한다. Simple은 정답, Reason은 reasoning과 정답을 학습한다. Instruct/Simple은 답변만, Reason은 reasoning과 답변을 요청하며 split·이미지·greedy 생성·최대 생성 길이를 맞춘다.

| 수정 조건 | 기본값 |
| --- | --- |
| 기반 모델 | Qwen2.5-VL-3B-Instruct 계열 4bit, backend별 변환본과 commit revision 고정 |
| epoch / batch / 누적 | 3 / 1 / 12 |
| 학습률 / weight decay | 2e-4 / 0.01 |
| LoRA | 언어 MLP gate/up/down projection, rank 8 / alpha 8 / dropout 0 |
| split seed | 3407 |
| 이미지 / 최대 학습 길이 | 6장 각각 224×224 / 2048토큰 |
| 생성 | greedy, 최대 512토큰 |
| loss 정책 | CUDA `train_on_responses_only=False`, MLX `train_on_completions=False` |
| 모델 선택 | epoch별 validation, 최저 validation loss, 2회 미개선 early stopping |

False는 질문·지시문 텍스트도 loss에 포함한다. CUDA collator는 image·padding 토큰을 별도 마스킹하지만, MLX-VLM 0.7.8의 full-sequence loss는 attention mask가 1인 이미지 placeholder도 포함한다. 이 차이는 `vision_loss_policy`로 기록한다. 논문이 명시한 마스킹 규칙이라고 단정한 것이 아니라 공개 CUDA 구현의 방향과 사용자 선택을 따른 것이다. 기반 vision·attention을 학습한다는 뜻은 아니다.

CUDA 단독 학습 기본 모델도 3B다. 모델과 processor를 같은 고정 snapshot에서 읽고 config·hash·revision·패키지·backend 정보를 저장한다. CUDA는 bitsandbytes 4bit·paged AdamW 8bit·선형 스케줄·warmup 50, MLX는 MLX 4bit·AdamW·고정 학습률·warmup 없음을 사용한다. 두 backend의 loss·최적화 조건이 같거나 논문 수치가 정확히 재현됐다고 주장하지 않는다.

수정 split의 4,218/534/528문항은 train/validation/test 할당 수다. 길이 제한 등으로 제외된 행은 실행 기록에서 확인하며, 할당 수와 optimizer가 실제 학습한 행 수를 구분한다. 시험 adapter는 최종 평가에서 거부한다.

</details>

<details>
<summary>논문 기반 reasoning 생성과 실험용 API 확장</summary>

`scripts/generate_reasoning.py`와 `scripts/reasoning_providers.py`는 논문 Table 1의 perception·prediction·planning·behavior별 prompt에 이미지 6장·질문·정답을 넣어 reasoning을 생성한다. 기존 정답과 원본 metadata를 보존한다.

- 논문의 GPT-4o 생성 절차: **논문 기반 재구현**.
- OpenAI/Gemini provider·모델 선택, 재시도·재개·운영 기록: **실험용 확장·설정 변경**.
- Gemini 생성 데이터는 논문의 GPT-4o 조건과 구분한다.
- 키는 환경변수로 받고 결과에 저장하지 않는다. 실패·미완료 행을 성공 데이터로 처리하지 않는다.
- GPT 채점은 구현하지 않았으며, 자동 학습 실험에서 생성 API를 호출하지 않는다.

</details>

<details>
<summary>공통 noGPT 평가 지표와 해석 범위</summary>

공개 저장소가 안내하던 `scripts/evaluation.py`는 없었다. 현재 평가기는 공식 DriveLM QA tag·좌표 그래프를 참고한 재구현이며, 저자의 비공개 코드와 완전히 같다고 보장하지 않는다.

| 지표 | 대상과 구현 |
| --- | --- |
| Accuracy | tag 0 문자열 일치 |
| BLEU-1~4 / ROUGE-L / CIDEr | tag 2, 공식 COCO backend |
| Match_noGPT | tag 3 좌표 매칭 F1 |
| 제외 범위 | GPT 전용 tag 1, reasoning 자체, 별도 추론한 그래프 시작 질문 |

```text
Language = (mean(BLEU-1~4) + ROUGE-L + CIDEr/10) / 3
Match_noGPT = 좌표 매칭 F1
Final_noGPT = (Accuracy + Language + Match_noGPT) / 3
```

Match_noGPT/Final_noGPT는 논문의 GPT 포함 Match/Final Score와 직접 비교할 수 없다. 예측 객체 그래프에 따라 채점 대상 수가 달라질 수 있어 counts와 그래프 제외 수를 함께 확인한다. 실패 행을 삭제해 비교하지 않으며 해당 지표의 대상이 없으면 N/A로 표시한다.

</details>

<details>
<summary>검증 상태</summary>

**2026-10-27 코드 검증**

- 전체 단위·회귀 테스트 **74개 통과**. 기존 50개에 공개 코드 연결부 9개, 공통 채점·보고서 7개, 실험 실행·결과 무결성 8개를 추가했다.
- 공개 Simple 확장 범위, 원본 파일 hash, 독립 학습 호출, 원본 분할과 학습·평가 중복 집계, 실패 행 보존, 공통 채점 연결, 완료 결과 집계와 변조 검사를 확인했다. GPU 호출과 언어 점수 backend를 대체한 테스트를 포함한다.
- 두 실험의 `--dry-run` 실행 계획과 셸 구문 검사를 통과했다. 실제 GPU 학습·추론이나 성능 점수 측정을 수행한 결과는 아니다.

아래는 **2026-10-26의 이전 검증 기록**이다. 당시 50개 테스트는 현재 전체 74개에 포함된다. 기록 파일은 `reports/1~3주차/validation_20261026.json`이다.

- 단위·회귀 테스트 50개 통과: 생성기 mock 17개, strict parser·graph 문맥·비교 조건·Instruct 무학습 분기·기존 입력 처리 등을 포함한다.
- Python 20개 구문 검사, 기존 실행 셸 2개 `bash -n`, `git diff --check` 통과. 당시 README는 작업 전 HEAD와 같았다.
- 실제 ReasonDrive test 528문항과 복원 context 5개에 정답을 예측으로 넣은 오프라인 평가 경로 검사 통과. dataset hash와 test 행 번호를 유지했고 그래프 제외는 0개였다. 언어 backend는 stub이므로 모델 추론·언어 점수의 검증 결과가 아니다.
- 해당 검사 대상은 Accuracy 205개, language 75개, coordinate match 27개, GPT-only 제외 221개였다. 실제 모델 응답의 채점 대상 수는 그래프 필터에 따라 달라진다.
- 키 없는 CLI dry-run에서 이미지 6개 입력, API 호출 0회, 생성 데이터 미작성을 확인했다. mock 외 실제 provider 연결은 실행하지 않았다.

**두 profile의 전체 GPU 학습·추론은 아직 실행하지 않았다.** CUDA/MLX 성능·메모리·속도, 유료 OpenAI/Gemini 호출, 논문 성능 재현은 확인하지 않은 상태다.

설치 패키지 목록은 실행별 `requirements-resolved.txt`로 저장한다. CUDA requirements는 호환 범위이며 실기 검증된 고정 lockfile이 아니다. 동일 패키지 목록도 GPU·driver·CUDA 환경의 동등성을 보장하지 않는다. 시험 학습 성공을 전체 학습·최장 샘플·성능 재현 성공으로 해석하지 않는다.

과거 로그의 4,199/529행, completion-only loss, 시험 학습 peak 메모리, 테스트 17/19개 통과 기록은 현재 조건의 검증으로 재사용하지 않는다. 기존 checkpoint의 데이터 중복·loss 조건도 소급 변경되지 않는다.

</details>

<details>
<summary>원본 보존·출처·공개 issue</summary>

공개 원본 기준은 [zhourobotics/reasondrive](https://github.com/zhourobotics/reasondrive)의 commit `326caecafedad1b9dbc156ebd77c8dc0eb998740`이다. 이전 C1 수정 전 로컬 기준은 `e38288ab3c59c02c1b28377ef27067c2b8104db6`이며, 이 문서는 코드 구현 상태와 실험 조건을 기록한다.

| 보존 자료 | 루트 기준 상대경로 |
| --- | --- |
| 원본 학습 코드 | `vendor/public_reasondrive/vlm-finetuning.py` |
| 원본 추론 코드 | `vendor/public_reasondrive/eval_finetuned_models.py` |
| 원본 라이선스 | `vendor/public_reasondrive/LICENSE` |
| 원본 commit·파일별 SHA-256 manifest | `vendor/public_reasondrive/manifest.json` |
| DriveLM 그래프 알고리즘·라이선스 | `scripts/vendor/drivelm_graph.py`, `scripts/vendor/LICENSE-DriveLM` |

공개 원본 파일은 보존하고 실행 연결부에서 필요한 확장을 적용한다. manifest로 원본 hash를 확인한다. DriveLM 그래프 기준 commit은 `1de72a74b257e5373400fa68239e99bd5d20580a`, language-evaluation은 `8e9e03feb7f5e605f20fc0c8dba0e24532a26216`이다. 데이터·이미지·모델 revision과 출처도 실행 기록에 남긴다.

- [#1 runtime comparison](https://github.com/zhourobotics/reasondrive/issues/1): 실행 시간 비교 문의. 이번에는 속도를 측정하지 않았다.
- [#2 evaluation error](https://github.com/zhourobotics/reasondrive/issues/2): 누락된 평가기·낮은 Accuracy 문의. 모든 게시 환경의 문제를 해결했다고 단정하지 않는다.
- [#3 model loading](https://github.com/zhourobotics/reasondrive/issues/3): 공개 모델 loader 예제 문제. README 예제는 수정하지 않았고 실행 코드는 VLM용 loader를 사용한다.

이슈의 존재는 논문 결과가 틀렸다는 증거가 아니다. 확인한 코드 차이, 선택한 실험 조건, 실제 실행 결과와 미검증 항목을 구분한다.

</details>
