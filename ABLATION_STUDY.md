# ReasonDrive 통제 비교 실험

이 실험의 출발점은 **사용자의 최종 수정본**인 `outputs/report_experiments/modified/20261028_180046_j4kzqit2`다. 이 수정본에서 Reason의 낮은 행동 예측 성능이 학습 배치, 손실 계산 대상, 추론 지시문·생성 방식과 관련되는지 확인한다. 실행 계획과 원시 결과는 `outputs/ablation_study/20261102_controlled/`에 저장한다. 아직 실행 중인 조건의 성능을 결과로 보고하지 않는다.

| 최종 수정본 기준 결과 | Simple | Reason |
| --- | ---: | ---: |
| Accuracy | 0.2917 | 0.1892 |
| Final_noGPT | 0.2560 | 0.2129 |
| 행동 예측 정답 | 19/70 | 10/70 |
| 정지 문항 정답 | 14/16 | 4/16 |

새 후보는 최종 수정본에 저장된 `training_config.json`의 실제 설정에서 파생한다. 해당 설정과 고정 분할을 유지하고 비교할 요인만 바꾼다. 현재 실행도 이 수정본과 같은 기반 모델 revision·데이터·장면 분할을 사용한다. 과거 공개 코드 실행의 학습/test 중복은 별도 감사 결과이며, 장면이 분리된 최종 수정본 내부의 Simple/Reason 성능 차이를 설명하는 원인으로 사용하지 않는다.

## 비교 조건

| 조건 | 모델 | microbatch × 누적 | 손실 계산 대상 |
| --- | --- | --- | --- |
| 기존 완료 모델 | Reason | 30 × 1 | 질문 + assistant 응답 |
| reason_b12_full | Reason | 12 × 1 | 질문 + assistant 응답 |
| simple_b12_full | Simple | 12 × 1 | 질문 + assistant 응답 |
| reason_b12_response | Reason | 12 × 1 | assistant 응답 |
| simple_b12_response | Simple | 12 × 1 | assistant 응답 |
| reason_b30_response | Reason | 30 × 1 | assistant 응답 |

이미지·padding 토큰은 모든 조건에서 손실에서 제외한다. Reason의 assistant 응답에는 추론 문장과 최종 답이 모두 포함된다. `response`는 최종 답에만 손실을 주는 설정이 아니다. 배치 변경은 optimizer 업데이트 수와 warmup 비율도 바꾸므로, 이를 배치 크기만의 효과로 해석하지 않는다.

**최종 수정본 Reason 가중치**로 현재/공개 지시문 × greedy/sampling의 네 조합을 먼저 검증한다. 파일 이름의 `public`은 지시문 문구만 뜻한다. 공개 코드로 학습한 모델을 다시 평가하는 것이 아니다. 새로운 다섯 학습은 같은 초기 가중치·장면 분할·seed에서 시작한다. 기존 완료 가중치에 추가 학습하는 방식은 학습량이 달라지므로 이 비교에 섞지 않는다. 학습 조건은 실행 전에 고정했으며, 검증 결과로 Reason 후보를 선정한 뒤 사전 지정한 다섯 모델을 최종 평가한다. 기존 test 결과는 이미 열람했으므로 새로운 외부 검증 자료로 취급하지 않는다.

## 지표 읽는 법

- `REPORT.md`: 실행 상태와 각 조건의 완료된 점수.
- `ANALYSIS.md`, `analysis.json`: 문항 유형별 결과, 학습 데이터의 최빈 답 기준선, 같은 문항에서 정답/오답이 바뀐 수. 분석 스크립트를 실행하면 갱신된다.
- `plan.json`: 데이터·코드·설정의 SHA256와 고정 계획.
- `logs/`, `models/`, `evaluations/`: 학습 로그, 모델, 생성 응답과 채점 결과.
- `../audit_current/`: 기존 실행 감사와 collator 마스크 확인 근거.

검증용 고정 tag0 정답률은 tag0 문항을 모두 채점한 진단 지표다. 그래프에 따라 채점 대상이 달라지는 논문식 Accuracy와 같지 않다. 정답 분포가 치우친 유형도 있으므로 행동 예측과 정지 판단을 따로 확인한다. `Final_noGPT` 역시 논문의 GPT 포함 Final과 다르다. 논문 수치에 도달하더라도 평가 문항이 다르면 정확히 같은 벤치마크를 재현했다고 표현하지 않는다.

## 실행과 복구

아래 명령은 저장된 계획을 이어서 실행한다. 이미 실행 중이면 중복 실행하지 않는다. `runner.lock`의 잠금으로 중복 실행을 막는다.

```bash
cd /home/osy/OSY_STUDY/261027/reasondrive-study
.venv/bin/python -u scripts/ablation_study.py \
  --run_dir outputs/ablation_study/20261102_controlled
```

같은 조건의 optimizer·scheduler·RNG checkpoint가 있으면 이어서 학습한다. 첫 checkpoint 전에 중단된 출력은 출처를 검증해 `failed_attempts/`에 보존한 뒤 같은 조건으로 처음부터 시작한다. 완료 모델과 원본 snapshot은 변경하지 않는다. 학습 도중 배치나 코드를 바꾸려면 새 실험으로 구분해야 한다.

GPU 실행 없이 보고서만 갱신하려면:

```bash
.venv/bin/python scripts/ablation_study.py \
  --run_dir outputs/ablation_study/20261102_controlled --report_only
.venv/bin/python scripts/analyze_ablation_results.py \
  --run_dir outputs/ablation_study/20261102_controlled
```

고정 snapshot은 실행을 시작할 때의 코드다. 이후 live 스크립트에 추가한 재개·출처 검증은 snapshot을 소급해 바꾸지 않는다. 과거 응답에 새 검증 항목이 없으면 그 한계를 기록하며, 과거부터 검증돼 있었다고 표시하지 않는다.
