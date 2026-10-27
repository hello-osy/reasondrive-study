# 공개 ReasonDrive 코드 보존본

저장소: https://github.com/zhourobotics/reasondrive

기준 commit: `326caecafedad1b9dbc156ebd77c8dc0eb998740`

두 Python 파일과 LICENSE는 해당 commit의 원문 그대로다. `manifest.json`에 파일 SHA-256을 기록한다.
원문 파일에는 주석이나 동작 변경을 추가하지 않는다.

실험용 확장·설정 변경은 `scripts/public_baseline.py`에서 수행한다.
Simple 모드와 실행 라이브러리의 인수명 호환을 추가하지만, 원본의 QA 분할·모델 선택·길이 불일치·샘플링 조건은 보존한다.
공통 noGPT 평가와 결과 수집은 `scripts/report_experiment.py` 및 `scripts/evaluate.py`에서 처리한다.
