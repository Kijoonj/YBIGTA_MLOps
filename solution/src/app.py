import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solution.src.features import request_to_dataframe


PRODUCTION_MODEL_DIR = ROOT_DIR / "solution" / "models" / "production_model"
LOG_PATH = ROOT_DIR / "solution" / "logs" / "predictions.csv"

app = FastAPI(title="MLOps Model Serving API")

model = None
metadata = {}


class PredictRequest(BaseModel):
    features: Dict[str, float]


@app.on_event("startup")
def load_model():
    global model, metadata

    model = mlflow.pyfunc.load_model(str(PRODUCTION_MODEL_DIR))
    metadata = json.loads((PRODUCTION_MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Production model is not loaded.")

    try:
        input_df = request_to_dataframe(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction = float(model.predict(input_df)[0])
    log_prediction(prediction)

    return {
        "prediction": prediction,
        "model_run_id": metadata.get("run_id"),
    }


def log_prediction(prediction):
    """예측 요청 결과를 csv 파일에 누적 저장합니다."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "model_run_id", "prediction"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_run_id": metadata.get("run_id"),
                "prediction": prediction,
            }
        )
