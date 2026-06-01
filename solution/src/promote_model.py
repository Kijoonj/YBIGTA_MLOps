import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PRODUCTION_MODEL_DIR = ROOT_DIR / "solution" / "models" / "production_model"


def parse_args():
    parser = argparse.ArgumentParser(description="Promote an MLflow run model to production.")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def write_metadata(run):
    metadata = {
        "run_id": run.info.run_id,
        "model_type": run.data.params.get("model_type"),
        "train_data_path": run.data.params.get("train_data_path"),
        "train_data_sha256": run.data.params.get("train_data_sha256"),
        "metric_name": "rmse",
        "metric_value": run.data.metrics.get("rmse"),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = PRODUCTION_MODEL_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def promote_to_production(run_id):
    """MLflow run의 model artifact를 현재 운영 모델 위치로 복사합니다."""
    mlflow.set_tracking_uri((ROOT_DIR / "mlruns").as_uri())
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    downloaded_model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model",
    )

    if PRODUCTION_MODEL_DIR.exists():
        shutil.rmtree(PRODUCTION_MODEL_DIR)

    shutil.copytree(downloaded_model_path, PRODUCTION_MODEL_DIR)

    metadata_path = write_metadata(run)
    return metadata_path


def main():
    args = parse_args()
    metadata_path = promote_to_production(args.run_id)

    print("Production model promoted")
    print(f"run_id: {args.run_id}")
    print(f"model_dir: {PRODUCTION_MODEL_DIR}")
    print(f"metadata_path: {metadata_path}")


if __name__ == "__main__":
    main()
