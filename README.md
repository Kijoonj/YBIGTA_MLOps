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
│   └── check_hw.py
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

MLflow UI 실행:

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

MLflow UI에서 RMSE가 가장 낮은 run id를 선택합니다.

아래 명령어를 실행하면 선택한 run의 `model` artifact가 `solution/models/production_model/`로 복사됩니다. 이 코드는 이미 구현되어 있습니다.

```bash
python solution/src/promote_model.py --run-id <BEST_RUN_ID>
```

성공 기준:

- `solution/models/production_model/` 디렉터리가 생성됩니다.
- `solution/models/production_model/metadata.json`이 생성됩니다.
- metadata에 run id, model type, RMSE, promoted time이 기록됩니다.

## PHASE 4: Production Model API 서빙

FastAPI 서버는 `solution/models/production_model/`의 모델과 `metadata.json`을 자동으로 로드합니다. 이 코드는 이미 구현되어 있습니다.

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

예측 요청:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"trip_distance": 3.2, "passenger_count": 1, "pickup_hour": 14, "is_weekend": 0}}'
```

성공 기준:

- prediction 값이 반환됩니다.
- 응답에 `model_run_id`가 포함됩니다.
- `solution/logs/predictions.csv`가 생성됩니다.

## PHASE 5: Drift Report 생성

이 단계는 실행만 하면 됩니다.

```bash
python solution/src/monitor.py
```

생성 파일:

```text
solution/reports/drift_report.txt
```

## PHASE 6: 최종 확인

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
[PASS] production model directory exists
[PASS] production metadata exists
[PASS] FastAPI health check returns 200 and model_loaded=true
[PASS] prediction log exists
[PASS] drift report exists

RESULT: PASS
```

## 제출물

다음 항목을 제출하세요.

자동 확인 결과
   - `python scripts/check_hw.py` 실행 결과 스크린 샷
