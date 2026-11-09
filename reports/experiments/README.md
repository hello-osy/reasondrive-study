# Git에 공유하는 실험 결과

모델 가중치·이미지·원본 QA·생성 전문은 `outputs/`에 보존하고 Git에서 제외한다. 이 폴더에는 작은 집계 표와 재현용 출처 해시만 저장한다.

- [원본 추론 vs Qwen 로컬 추론 Reason 비교](local_reasoning_20261109/comparison.md)

프로젝트 루트에서 결과표를 다시 추출하는 명령:

```bash
.venv/bin/python scripts/export_reasoning_comparison.py \
  --run outputs/local_reasoning_experiments/Qwen2.5-VL-3B-Instruct-bnb-4bit_reason_drivelm_20261109_151821 \
  --output reports/experiments/local_reasoning_20261109
```

로컬 원본 평가 파일이 필요하다. 공유된 표만 열어보는 데에는 모델이나 데이터가 필요하지 않다.
