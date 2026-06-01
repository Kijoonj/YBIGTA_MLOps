# MLflow 기반 Mini MLOps Pipeline 실습 과제

## 과제 개요

실제 운영 환경에서는 다음 질문에 답할 수 있어야 합니다.

- 이 모델은 어떤 run에서 만들어졌는가?
- 어떤 데이터 파일로 학습했는가?
- 같은 파일명이라도 데이터 내용이 바뀌었는지 확인할 수 있는가?
- 어떤 parameter와 metric을 가진 모델인가?
- 지금 API 서버가 사용하는 production model은 무엇인가?

이번 실습은 위 질문에 답하기 위한 가장 작은 MLOps 구조를 만듭니다.

## 사전 준비

Python 3.10 이상을 권장합니다.

macOS / Linux / WSL2:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

설치 확인:

```bash
python --version
mlflow --version
uvicorn --version
```

## 기본 규칙

- `baseline/` 디렉터리는 수정하지 마세요.
- `scripts/` 디렉터리는 수정하지 마세요.
- 학생이 수정할 파일은 `solution/src/train.py` 하나입니다.
- `solution/src/train.py` 안의 `____` 빈칸만 수정하세요.
- 제출 전 FastAPI 서버를 켠 상태에서 `python scripts/check_hw.py`를 실행하세요.

## 폴더 구조

```text
mlops-mlflow-hw/
├── baseline/
│   └── train_baseline.py
├── data/
│   ├── train.csv
│   ├── train_v1.csv
│   ├── train_v2.csv
│   ├── train_shifted.csv
│   ├── valid.csv
│   └── new_data.csv
├── scripts/
│   ├── check_hw.py
│   └── send_new_data.py
├── solution/
│   ├── src/
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── promote_model.py
│   │   ├── app.py
│   │   └── monitor.py
│   ├── models/
│   ├── reports/
│   └── logs/
├── requirements.txt
└── README.md
```

## 제공 데이터셋

모든 학습 데이터셋은 같은 컬럼 구조를 가집니다.

| 파일 | 의미 |
|---|---|
| `data/train_v1.csv` | 기본 학습 데이터 |
| `data/train_v2.csv` | v1에 최신 샘플이 추가된 데이터 |
| `data/train_shifted.csv` | 긴 이동 거리와 주말/야간 요청이 많은 shifted 데이터 |
| `data/valid.csv` | 공통 validation 데이터 |
| `data/new_data.csv` | drift report용 신규 데이터 |

## PHASE 1: Baseline 확인

먼저 기존 baseline 학습 코드를 실행합니다.

```bash
python baseline/train_baseline.py
```

확인할 점:

- metric은 출력되지만 실험 기록으로 남지 않습니다.
- 모델 파일은 저장되지만 어떤 데이터/parameter로 만들어졌는지 추적하기 어렵습니다.
- production model이라는 명확한 운영 모델 개념이 없습니다.

이 단계는 수정하지 않습니다.

## PHASE 2: MLflow 기록 빈칸 채우기

`solution/src/train.py`의 `log_experiment_to_mlflow()` 안에 있는 `____` 빈칸을 채웁니다.

이미 제공되는 코드:

- `--train-data`, `--valid-data` 인자로 데이터 선택
- 데이터 로드
- feature/target 분리
- 모델 생성
- 모델 학습
- RMSE, MAE, R2 계산
- 데이터 SHA256 hash 계산
- feature artifact 파일 생성

TODO:

- `mlflow.log_param(...)`의 값 빈칸 채우기
- `mlflow.log_metric(...)`의 metric key 빈칸 채우기
- `mlflow.log_artifact(...)`의 파일 경로 빈칸 채우기
- `mlflow.sklearn.log_model(...)`의 모델과 artifact path 빈칸 채우기

예시 형태:

```python
mlflow.log_param("model_type", ____)
mlflow.log_param("train_data_sha256", sha256_file(____))
mlflow.log_metric("rmse", metrics["____"])
mlflow.sklearn.log_model(____, artifact_path="____")
```

MLflow에 반드시 기록해야 하는 값:

```text
model_type
n_estimators
learning_rate
random_state

train_data_path
valid_data_path
train_data_sha256
valid_data_sha256
train_row_count
valid_row_count

rmse
mae
r2
```

서로 다른 데이터 / 모델 타입을 적용해 최소 4개의 run을 만드세요.

macOS / Linux / WSL2:

```bash
python solution/src/train.py \
  --model-type random_forest \
  --run-name rf_train_v1 \
  --train-data data/train_v1.csv \
  --valid-data data/valid.csv

python solution/src/train.py \
  --model-type random_forest \
  --run-name rf_train_v2 \
  --train-data data/train_v2.csv \
  --valid-data data/valid.csv

python solution/src/train.py \
  --model-type random_forest \
  --run-name rf_train_shifted \
  --train-data data/train_shifted.csv \
  --valid-data data/valid.csv

python solution/src/train.py \
  --model-type gradient_boosting \
  --run-name gb_train_v2 \
  --train-data data/train_v2.csv \
  --valid-data data/valid.csv
```

Windows PowerShell:

```powershell
python solution/src/train.py --model-type random_forest --run-name rf_train_v1 --train-data data/train_v1.csv --valid-data data/valid.csv
python solution/src/train.py --model-type random_forest --run-name rf_train_v2 --train-data data/train_v2.csv --valid-data data/valid.csv
python solution/src/train.py --model-type random_forest --run-name rf_train_shifted --train-data data/train_shifted.csv --valid-data data/valid.csv
python solution/src/train.py --model-type gradient_boosting --run-name gb_train_v2 --train-data data/train_v2.csv --valid-data data/valid.csv
```

이후 MLflow UI 실행하면 위에서 학습시킨 모델들의 로그를 확인할 수 있습니다:

```bash
mlflow ui --backend-store-uri ./mlruns
```

접속 주소:

```text
http://127.0.0.1:5000
```

성공 기준:

- MLflow UI에 최소 4개의 run이 보여야 합니다.
- 최소 3개 이상의 서로 다른 `train_data_path`가 기록되어야 합니다.
- 최소 2개 이상의 서로 다른 `model_type`이 기록되어야 합니다.
- 각 run에 `train_data_sha256`, `valid_data_sha256`이 기록되어야 합니다.
- artifact 안에 `feature_columns.json`과 `model`이 있어야 합니다.

## PHASE 3: Production Model 승격

위에서 우리가 학습시킨 모델 중 가장 좋은 것을 선택해 프로덕션 모델로 승격시켜 봅시다. MLFlow에서 모델 중 RMSE가 가장 낮은 모델을 찾아 run id를 선택합니다.

아래 명령어를 실행하면 선택한 run의 model artifact가 solution/models/production_model/로 복사됩니다. 이 코드는 이미 구현되어 있습니다.

```bash
python solution/src/promote_model.py --run-id <BEST_RUN_ID>
```

성공 기준:

- `solution/models/production_model/` 디렉터리가 생성됩니다.
- `solution/models/production_model/metadata.json`이 생성됩니다.
- metadata에 run id, model type, 학습 데이터 경로, 학습 데이터 hash, RMSE, promoted time이 기록됩니다.

## PHASE 4: Production Model API 서빙과 Drift Report 확인

방금 우리가 선정한 프로덕션 모델을 FastApi서버를 사용해 서빙됩니다. 서버의 코드는 이미 구현되어 있습니다. 이제 이 서버로 예측요청을 보내보고 마지막으로 Drift에 대해 실습 해봅시다.

수업에서 다루었듯, Drift는 모델이 학습할 때 본 데이터 분포와 운영 중 API로 들어오는 데이터 분포가 달라지는 현상입니다. 예를 들어 학습 데이터에는 짧은 이동 거리가 많았는데, 운영 요청에는 긴 이동 거리나 주말 요청이 많아지면 모델이 익숙하지 않은 입력을 받게 됩니다. 이런 경우에는 모델 성능이 떨어질 수 있으므로, 요청의 feature 분포를 학습 데이터와 비교해 확인해야 합니다.

이 과제에서는 `metadata.json`에 기록된 `train_data_path`를 기준 학습 데이터로 사용합니다. 서버가 `/predict` 요청을 받을 때마다 요청 feature와 prediction을 `solution/logs/predictions.csv`에 누적 저장하고, 이 요청 로그를 기준 학습 데이터와 비교해 `solution/reports/drift_report.txt`를 갱신합니다. 각 feature의 평균 차이를 학습 데이터의 표준편차로 나눈 값을 `drift_score`로 계산하고, 기준값 이상이면 drift가 발생한 것으로 표시합니다.

서버 실행:

```bash
uvicorn solution.src.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

예상 응답:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

학습 데이터와 다른 분포의 신규 데이터셋을 prediction API로 여러 건 요청하고, drift report 경로를 확인합니다.

FastAPI 서버가 켜져 있는 상태에서 다른 터미널에서 실행하세요.

```bash
python scripts/send_new_data.py --data data/new_data.csv
```

이 스크립트는 `data/new_data.csv`의 각 row를 `/predict`에 보냅니다. API는 `metadata.json`의 `train_data_path`를 기준 데이터로 사용하고, 현재 production model의 `model_run_id`와 일치하는 prediction 요청 로그를 비교해 drift report를 갱신합니다.

예상 출력:

```text
sent_rows: 10
last_prediction: <PREDICTION>
model_run_id: <RUN_ID>
drift_report_path: <DRIFT_REPORT_PATH>
```

필요하면 같은 로직을 직접 실행해 report만 다시 생성할 수도 있습니다.

```bash
python solution/src/monitor.py
```

생성/갱신 파일:

```text
solution/logs/predictions.csv
solution/reports/drift_report.txt
```

Drift report에는 다음 정보가 포함되어야 합니다.

- `DRIFT_CHECK`
- `reference_train_data_path`
- `request_count`
- feature별 `drift_score`

## PHASE 5: 최종 확인

FastAPI 서버가 켜져 있는 상태에서 아래 명령어를 실행합니다.

```bash
python scripts/check_hw.py
```

예상 출력:

```text
[PASS] mlruns directory exists
[PASS] at least 4 MLflow runs found
[PASS] at least 3 different training datasets logged
[PASS] at least 2 different model types logged
[PASS] train data sha256 logged for each experiment
[PASS] valid data sha256 logged for each experiment
[PASS] training row count logged for each experiment
[PASS] production model directory exists
[PASS] production metadata exists
[PASS] production metadata has required keys
[PASS] production metadata values are not empty
[PASS] FastAPI health check returns 200 and model_loaded=true
[PASS] prediction log exists
[PASS] prediction log has required columns
[PASS] prediction log includes feature columns
[PASS] drift report exists
[PASS] drift status calculated
[PASS] drift score included in report
[PASS] drift report includes reference training data
[PASS] drift report includes request count

RESULT: PASS
```

## 제출물

다음 항목을 제출하세요.

`python scripts/check_hw.py` 실행 결과 스크린 샷
