# 복붙 스크립트

## M4 MacBook Pro

```bash
cd /Users/hello-osy/OSY_STUDY/261017/ReasonDrive
bash scripts/run_mac.sh
```

- 경로 입력 없이 **학습 JSON + 필요한 이미지 자동 다운로드·검증**.
- 기본 저장: `data/drivelm.json`, `data/nuscenes/`. 기존 파일은 재사용하고 손상·누락 이미지만 복구.
- 이미지 train ZIP 약 **3.48 GB**, 모델 가중치 약 **3.07 GB**. 추가 이미지가 필요하면 val ZIP 약 0.70 GB도 다운로드. 압축 파일·환경·모델을 포함해 디스크 여유 필요.
- 공식 Hugging Face → 접근 불가 시 공식 Google Drive 미러 순으로 시도.
- 제공된 최신 실행 로그에서 데이터·모델 다운로드 완료 확인. 재실행 시 재사용.
- 다른 환경에서 HF 403 발생 시 [공식 데이터 페이지](https://huggingface.co/datasets/OpenDriveLab/DriveLM) 접근 승인 필요. 두 다운로드 경로가 실패하면 설명 후 종료.

- 실행 순서: 환경 설치 → 데이터 다운로드/검증 → 데이터 준비 → 시험 학습 → 두 모델 순차 학습 → test 추론 → 비교 표.
- 결과: `outputs/mac_실행시간/comparison/`의 Markdown·CSV·JSON.
- 가속 시도(3 epoch 유지, 메모리 사용 증가): `GRAD_CHECKPOINT=0 bash scripts/run_mac.sh`
- 빠른 예비 비교(두 모델 모두 1 epoch): `EPOCHS=1 GRAD_CHECKPOINT=0 bash scripts/run_mac.sh`. 3 epoch 실험과 학습량이 다름.
- checkpointing 해제 시험에서 메모리 부족 시 `GRAD_CHECKPOINT=1 bash scripts/run_mac.sh`로 실행.
- 실행 중인 작업에는 변경이 적용되지 않음. 적용하려면 해당 터미널에서 Ctrl+C 후 재실행. 기존 출력은 보존하지만 학습 가중치는 이어받지 않고 새로 시작.
- 전체 학습은 오래 걸릴 수 있음. 16GB에서 실제 전체 학습 완료는 아직 미검증이며, 시험 학습 실패 시 이후 단계 자동 중단.

## Ubuntu 24.04 · amd64 · RTX 5070 Ti (16GB)

네이티브 Ubuntu 24.04, **amd64 = x86_64** 기준입니다. 프로젝트 루트(`/home/osy/OSY_STUDY/261017/reasondrive-study`)의 `.venv` 하나에 Python 3.11 + Unsloth를 설치합니다.

**최초 1회: 시스템 도구 설치**

```bash
cd /home/osy/OSY_STUDY/261017/reasondrive-study
sudo apt update
sudo apt install -y build-essential git curl ca-certificates python3-venv openjdk-17-jre-headless ubuntu-drivers-common
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/reasondrive-install-uv.sh
  sh /tmp/reasondrive-install-uv.sh
fi
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi
```

`nvidia-smi`가 실패하면 드라이버부터 설치합니다. 이미 정상 작동하면 아래 단계는 생략합니다.

```bash
sudo ubuntu-drivers install
sudo reboot
```

재부팅 후 `nvidia-smi`에 RTX 5070 Ti가 표시되는지 확인합니다. Secure Boot를 사용한다면 설치 중 안내된 MOK 등록도 완료해야 합니다. 설치되는 PyTorch CUDA runtime과 호환되는 드라이버가 필요합니다.

**전체 실행: 그대로 복붙**

```bash
cd /home/osy/OSY_STUDY/261017/reasondrive-study
export PATH="$HOME/.local/bin:$PATH"
bash scripts/run_ubuntu.sh
```

- 환경 설치·의존성 검사 → CUDA/bitsandbytes 4bit forward/backward 점검 → JSON·이미지 자동 다운로드/검증 → Reason 시험 학습 1 optimizer step(12 microstep) → Simple/Reason 각각 새 기반 모델에서 학습 → 독립 test 추론 → 비교 표 생성.
- 기본값: **Qwen2.5-VL-3B bnb 4bit, 3 epoch, batch 1, 누적 12, LoRA 8/8, 이미지당 224×224, 최대 2048 토큰**, gradient checkpointing 사용. 16GB 여유를 고려한 시작 설정이며 OOM이 없음을 보장하지 않습니다.
- 두 모델은 같은 scene split을 사용합니다. 시험 학습 adapter는 재사용하지 않으며 최종 평가에서도 거부합니다. 시험 학습은 일부 배치의 실행 확인으로, 최장 샘플·전체 학습 메모리 검증은 아닙니다.
- validation batch도 1로 제한하고 epoch마다 validation·checkpoint 저장 후 validation loss 최저 모델을 사용합니다. 2회 미개선 시 early stopping.
- 결과: `outputs_ubuntu_날짜_시간_임의문자/comparison/comparison.md` 및 CSV·JSON. 실행 폴더의 `requirements-resolved.txt`와 `nvidia-smi.txt`에 설치 버전과 GPU/드라이버 정보를 보관합니다.
- 기존 데이터·모델 캐시는 재사용합니다. 재실행은 새 출력 폴더에서 처음부터 학습하며 checkpoint 재개는 하지 않습니다. W&B는 기본 비활성화입니다.
- 이미지 train ZIP 약 3.48GB 외에도 모델·CUDA 패키지·checkpoint를 위한 디스크 공간이 필요합니다. 모든 단계는 오류 발생 시 중단합니다.

```bash
# 빠른 예비 비교: 두 모델 모두 1 epoch
EPOCHS=1 bash scripts/run_ubuntu.sh

# 이미 받은 이미지 디렉터리를 사용
bash scripts/run_ubuntu.sh /absolute/path/to/nuscenes
```

**Blackwell 설치 및 구현 변경**

- RTX 50 계열을 위해 `uv pip install --torch-backend=auto`로 GPU에 맞는 PyTorch wheel을 선택합니다. CUDA 12.8 미만 wheel은 Blackwell 사전 점검에서 거부합니다. [Unsloth 공식 설치 안내](https://github.com/unslothai/unsloth), [PyTorch Blackwell 지원 안내](https://pytorch.org/blog/pytorch-2-7/).
- 시스템 CUDA toolkit을 별도로 설치하는 명령은 포함하지 않습니다. 드라이버가 선택된 CUDA runtime보다 오래된 경우에는 드라이버 업데이트가 필요할 수 있습니다.
- `requirements-cuda.txt`는 설치 시 호환 의존성을 해결하는 목록이며 **실기 검증된 고정 lockfile은 아닙니다**. 실제 설치 조합은 실행 폴더에 기록합니다.
- CUDA 코드를 최신 API 인자(`processing_class`, `max_length`, `eval_strategy`)에 맞추고, validation 메모리 제한·processor 저장·명시적 출력 경로·시험 학습 옵션을 추가했습니다. [TRL API 문서](https://huggingface.co/docs/trl/sft_trainer).
- Ubuntu 실행 기본 3B는 기존 CUDA 기본 7B와 다른 실험입니다. Mac MLX 실험과도 backend·optimizer 등이 다르므로 완전히 같은 조건의 재현은 아닙니다.
- **사용자가 제공한 호스트 로그에서 RTX 5070 Ti와 드라이버 580.178.04의 정상 인식을 확인했습니다.** 에이전트 컨테이너에서는 GPU에 접근할 수 없어 실제 CUDA 설치·학습 완료는 미검증입니다. 문법 검사 및 CPU 회귀 테스트 결과는 아래 검증 항목에 기록합니다.

# 변경 사항 설명

## Ubuntu 실행 로그의 skip·예시 추론 오류 수정

- `run_ubuntu.sh`와 `run_mac.sh` 모두 다운로드·이미지 복구 직후 공통 `prepare_inputs.py`를 자동 실행합니다. 별도 수동 복구 명령은 필요 없습니다.
- 실행 폴더의 `input_repair_report.json`에 split별 통과 수, 설명 정규화 행, 복구 불가 행을 기록합니다. 원본 JSON·행 번호·분할은 유지하고 실제 학습 입력에서 동일한 정규화를 적용합니다. 반복 실행해도 복구 결과는 동일합니다.
- 복구 불가 입력은 보고서를 저장하고 학습 전에 중단합니다. 빈 설명을 지어내거나 정답을 임의 수정하지 않습니다. Mac의 토큰 길이 검사는 이후 별도로 수행합니다.
- Mac 전처리 plan에 형식 버전을 저장해 이전 캐시는 다시 만들고, 새 버전은 정상 재사용합니다. CUDA 예시 추론에서 예상하지 못한 오류가 나면 traceback과 함께 중단하여 전체 학습으로 넘어가지 않습니다.

- 원본 `thinking`의 태그 밖 설명, 닫는 태그 누락, 반복 태그, `<thinkThe` 오타 때문에 제외되던 train 19개·validation 5개를 복구했습니다. 원문 JSON은 변경하지 않고, 설명·좌표·문장 순서를 보존한 채 학습 입력에서만 태그와 줄바꿈을 정규화합니다. 원문에 포함된 거절 문구나 설명의 사실성까지 교정하는 작업은 아닙니다.
- 정답 및 생성 출력의 `<answer>` 검증은 그대로 엄격하게 유지합니다. 빈 설명이나 누락 이미지는 여전히 오류로 처리합니다.
- 실제 5,280행 모두 Simple/Reason 형식·이미지 경로 검사 통과: train 4,218 / validation 534 / test 528. 토큰 길이·GPU 전체 학습 검증과는 별개이며 Mac 전처리에서는 길이 초과 행이 추가 제외될 수 있습니다. Mac의 이전 전처리 캐시는 자동 무효화합니다.
- Arrow Dataset이 텍스트 항목에 추가한 `image=None` 때문에 예시 추론에서 발생하던 `NoneType ... startswith` 오류를 재현하고 수정했습니다. 추론 전 null 필드를 제거하며, 로컬 이미지 URI의 `%2B`·`%20`도 실제 경로로 복원합니다.
- 예시 및 최종 test 추론에서 processor의 `images=`·`text=` 인자를 명시하고 chat template의 `tokenize=False`를 설정했습니다. 예시 추론 실패 시에는 전체 traceback을 남깁니다.
- 회귀 테스트 19개 통과. 실제 설치된 Dataset/Qwen 이미지 처리기로 오류 재현 후 이미지 6장(각 224×224) 처리 성공을 확인했습니다. GPU 생성 완료는 컨테이너에서 미검증입니다.
- 실행 중인 프로세스에는 반영되지 않습니다. 수정 적용은 기존 작업을 종료한 뒤 `bash scripts/run_ubuntu.sh`로 새 실행하며, 이전 학습을 이어받지는 않습니다.


## M4 MacBook Pro 16GB 대응

- 기존 `scripts/vlm-finetuning.py`는 CUDA/Unsloth 기반이므로 Mac에서 그대로 사용하지 않음. NVIDIA용 코드는 유지.
- `scripts/mac_pipeline.py` 추가: MLX 기반 `prepare` / `train` / `infer`.
- `scripts/run_mac.sh`: 환경 준비부터 비교 표까지 순차 실행. 인자 없이 자동 데이터 준비; 기존 이미지 경로를 첫 인자로 주는 방식도 유지.
- `scripts/download_data.py` 추가: QA 5,280개와 고유 이미지 4,800개 기준 자동 준비. 공식 archive 중 필요한 파일만 추출하고 ZIP CRC·이미지 형식·경로·디스크 여유 검사. 다운로드 출처 및 누락 현황을 `data/download_report.json`에 기록.
- 접근 실패 시 재로그인과 접근 승인을 구분해 안내하고, 예상된 다운로드 오류는 긴 traceback 없이 종료.
- JSON/이미지 저장소 revision 고정, 기존 정상 이미지·완료 ZIP 재사용, 지원되는 다운로드의 부분 파일 재개. Google Drive 미러용 `gdown==5.2.0` 추가.
- `requirements-mac.txt` 추가: MLX-VLM 및 평가 의존성. PyTorch/torchvision은 Qwen 전처리 의존성이고 학습 backend는 MLX.
- Mac 기본값: **Qwen2.5-VL-3B 4bit, 배치 1, 누적 12, LoRA rank/alpha 8/8, 이미지 6장 각각 224×224, 최대 2048 토큰**.
- 언어 MLP의 `gate_proj/up_proj/down_proj`에 LoRA 적용. 기반 모델·비전·attention 가중치 고정.
- gradient checkpointing, 데이터 캐시 제한, MLX 메모리 한도 10 GiB 적용. 이 한도는 프로세스 전체 메모리 제한이 아님.
- 먼저 Reason 모드의 가장 긴 학습 샘플로 12 microstep 시험 학습. MLX peak 9 GiB 초과 시 전체 학습 중단.
- Simple/Reason을 동시에 띄우지 않고 각각 새로운 기반 모델에서 순차 학습.
- validation loss 최저 adapter 저장, 2회 미개선 시 early stopping. 시험 학습 adapter는 최종 평가에서 사용 금지.
- Mac에서는 **출력 토큰만 CE 학습**. KL이나 teacher 모델은 추가하지 않음.
- 논문과 다른 부분: MLX 4bit 및 일반 AdamW, 고정 학습률 2e-4, warmup 없음. CUDA/bnb4·paged AdamW·선형 스케줄과 동일한 재현은 아님.
- 최대 길이를 넘거나 잘못된 train/validation 샘플은 **두 모드에서 동일하게 제외**하고 `data/mac_plan.json`에 기록. test는 부분 삭제하지 않고 이미지 누락 시 준비 단계 실패.

## 속도 개선

- 전체 validation 주기: 120 microstep마다 → 약 1 epoch마다 + 마지막 step. 현재 4,199 train 기준 4,200/8,400/12,597 step, 약 105회 → 3회. best adapter 선택·early stopping 시점도 달라짐.
- 검증된 전처리 plan 재사용: 데이터·모델·길이·이미지 경로/크기/수정 시각이 같으면 약 10분의 재전처리 생략. `prepare --rebuild`로 강제 재검사.
- `GRAD_CHECKPOINT=0`: 역전파 재계산을 줄이는 옵션. 기본값 1은 메모리 절약 유지. 해제 시에도 긴 샘플 시험 학습·9 GiB 검사 유지.
- `EPOCHS` 환경변수 추가(기본 3). 두 모델에 동일 적용.
- 제공 로그의 Simple 약 6.17초/step 기준 순수 학습만 약 21.6시간/3 epoch. validation 비용 감소와 별개이며 새 설정의 실제 가속 배율은 미측정.

## 공통 학습·데이터 수정

- `scripts/data_utils.py` 추가: 입력 포맷·태그 처리·JSON 저장·데이터 분할 공통화.
- `--mode simple`: 이미지+질문 → 정답만 학습.
- `--mode reason`: 이미지+질문 → 설명+정답 학습. 원본 Reason 학습은 이미 있었고 Simple 모드를 추가한 것.
- 동일 기반 모델을 각각 파인튜닝. Simple 모델을 추가 학습해 Reason 모델로 만드는 실험이 아님.
- 장면 ID 기준 train/validation/test를 약 80/10/10으로 분리. 동일 장면이 여러 split에 들어가지 않음.
- 분할 seed 기본 3407. SHA-256과 행 인덱스를 manifest에 저장하고 평가 때 재사용.
- validation만 early stopping에 사용. test는 최종 평가용으로 보존.
- 누락 이미지·잘못된 필드로 실패한 샘플을 기록하고 제외. `None`이 학습 리스트에 남는 문제 해결.
- 정상적인 여러 `<think>` 블록을 지원. `<answer>`는 한 개의 비어 있지 않은 태그 쌍 요구.
- CUDA 코드의 trainer/collator 최대 길이를 한 설정으로 통일.
- CUDA 코드에 `--no-use_wandb` 추가, 출력 폴더명에 모드 표기.
- 학습 후 예시 추론에서 입력 프롬프트 토큰을 제거하고 greedy 생성.

## 응답 생성 수정

- CUDA: `scripts/eval_finetuned_models.py`, Mac: `scripts/mac_pipeline.py infer`.
- 학습 설정·분할을 검증하고 저장된 독립 test만 사용. 평가 때 재분할하지 않음.
- 모델에는 이미지·질문만 입력. 정답·설명은 추론 입력에서 제외.
- 기본 greedy 생성. CUDA의 선택적 샘플링은 행별 seed 고정.
- 출력 폴더 자동 생성, 실패 행도 응답 JSON에 보존.
- 원본 응답·추출 답변·상태·오류 및 `.meta.json`의 데이터 hash/test 인덱스/생성 설정 저장.
- 서로 다른 GPU·라이브러리 환경 사이의 비트 단위 재현성까지 보장하지 않음.

## 비교 지표 수정

- 원본 README가 가리키던 `evaluation.py`는 없었음. `scripts/evaluate.py` 새로 추가.
- 처음 구현한 **로컬 정답 일치율·token F1·출력 형식 성공률은 사용자 요청으로 제거**.
- 현재 표: **Accuracy, BLEU-1~4, ROUGE-L, CIDEr, Match_noGPT, Final_noGPT**.
- 공식 DriveLM의 원본 QA tag와 좌표 그래프 필터 적용. 모든 질문에 모든 metric을 적용하지 않음.
- Accuracy: tag 0 문자열 완전 일치. 언어 지표: tag 2 공식 COCO backend. Match_noGPT: tag 3 좌표 F1.
- `<think>`는 채점하지 않고 `<answer>`만 평가. GPT API 호출 없음. GPT 전용 tag 1은 제외.
- 공식 Match에도 GPT 점수가 포함되어 있어 GPT 제외 점수는 별도 이름 사용.

```text
Language = (mean(BLEU-1~4) + ROUGE-L + CIDEr/10) / 3
Match_noGPT = 좌표 매칭 F1
Final_noGPT = (Accuracy + Language + Match_noGPT) / 3
```

- Match_noGPT/Final_noGPT는 논문 원래 Match/Final Score와 직접 비교 불가.
- 그래프 필터 때문에 모델별 실제 채점 질문 수가 달라질 수 있어 JSON `counts`에 기록. 대상 질문이 없으면 N/A.
- 누락/중복 행, 데이터 hash·정답·생성 설정 불일치, 불완전한 test 장면을 검증.
- `scripts/vendor/drivelm_graph.py`: 공식 좌표/그래프 코드 추출, Apache-2.0 라이선스 동봉.
- `requirements-eval.txt`: language-evaluation 커밋 고정. PTB tokenizer를 위한 Java 필요.
- DriveLM 기준 커밋: `1de72a74b257e5373400fa68239e99bd5d20580a`.
- language-evaluation 기준 커밋: `8e9e03feb7f5e605f20fc0c8dba0e24532a26216`.
- README 평가 안내 수정. `.gitignore`에 가상환경·캐시·실험 결과·분할 폴더 제외 규칙 추가.

## 검증 및 제한

- Ubuntu 추가 변경 후 CPU 회귀 테스트 17개 재통과, `bash -n` 및 Python 컴파일 검사 통과. 호스트의 `nvidia-smi`는 정상입니다. 에이전트 컨테이너의 GPU 접근 제한으로 실제 CUDA 실행은 수행하지 못했습니다.

- 자동 테스트 17개 통과: 공통 파이프라인 11개 + 다운로드 4개 + 전처리 캐시 무효화·validation 주기 2개. 셸/Python 문법 검사 통과.
- 사용자 최신 로그: 5,280 QA·4,800개 이미지 다운로드/추출 완료, 전처리 후 train 4,199 / validation 529 / test 528 (24개 제외).
- 실제 공식 언어 backend로 합성 응답의 BLEU/ROUGE-L/CIDEr 및 비교 표 생성 확인.
- Mac Metal 접근, 소형 임의 모델의 4bit LoRA 가중치 갱신·CE·gradient checkpointing·validation·adapter 저장/복원 확인.
- 실제 Qwen 전처리기에서 6장 이미지가 각각 224×224에 해당하는 `[1,16,16]` patch grid로 처리되고, 출력에만 loss mask가 적용되는 것 확인. 이 전처리 검사는 모델 가중치 없이 수행.
- 사용자 최신 로그: 실제 3B Reason 시험 학습 12 microstep 성공, MLX peak 약 5.48 GiB(checkpointing 켬). Simple 전체 학습 진입 확인. 전체 완료·두 모델 test 비교 및 checkpointing 해제 성능/메모리는 아직 미검증.
- 공개 코드에서 발견한 문제를 수정한 것이며 논문 저자의 실제 실험 오류를 입증한 것은 아님.
- 기존 checkpoint의 누출 문제는 소급 해결되지 않으므로 새 분할로 재학습 필요.
- 기존 `docker_run.sh`는 수정하지 않았습니다. Ubuntu 실행은 새 `scripts/run_ubuntu.sh`와 `requirements-cuda.txt`를 사용합니다.
