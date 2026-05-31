import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solution.src.features import FEATURE_COLUMNS, split_features_target


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model and log the experiment to MLflow.")
    parser.add_argument("--model-type", choices=["random_forest", "gradient_boosting"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-data", default="data/train_v1.csv")
    parser.add_argument("--valid-data", default="data/valid.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resolve_data_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def sha256_file(path):
    """데이터 파일 내용이 바뀌었는지 확인하기 위한 해시값을 계산합니다."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_model(args):
    if args.model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
        )

    return GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )


def evaluate_model(model, x_valid, y_valid):
    predictions = model.predict(x_valid)
    return {
        "rmse": mean_squared_error(y_valid, predictions) ** 0.5,
        "mae": mean_absolute_error(y_valid, predictions),
        "r2": r2_score(y_valid, predictions),
    }


def write_feature_artifact():
    feature_path = ROOT_DIR / "solution" / "reports" / "feature_columns.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    return feature_path


def log_experiment_to_mlflow(args, model, metrics, feature_path, train_df, valid_df):
    """핵심 TODO: 모델, 데이터, 성능 정보를 하나의 MLflow run으로 묶어 기록합니다."""
    mlflow.set_tracking_uri((ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment("taxi-fare-mlops")

    train_path = resolve_data_path(args.train_data)
    valid_path = resolve_data_path(args.valid_data)

    # TODO: 아래 정보를 MLflow run 하나에 기록하세요.
    #
    # 1. 모델/실험 parameter
    #    - model_type, n_estimators, learning_rate, random_state
    #
    # 2. 데이터 재현성 parameter
    #    - train_data_path, valid_data_path
    #    - train_data_sha256, valid_data_sha256
    #    - train_row_count, valid_row_count
    #
    # 3. 성능 metric
    #    - rmse, mae, r2
    #
    # 4. artifact
    #    - feature_columns.json
    #    - sklearn model을 artifact_path="model"로 저장
    #
    # 힌트: metric 기록을 위한 mlflow 문법입니다.
    # with mlflow.start_run(run_name=args.run_name) as run:
    #     mlflow.log_metric("rmse", metrics["rmse"])


    with mlflow.start_run(run_name=args.run_name) as run:
        # 1. 모델/실험 parameter 기록
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("random_state", args.random_state)

        # 2. 데이터 재현성 정보 기록
        mlflow.log_param("train_data_path", args.train_data)
        mlflow.log_param("valid_data_path", args.valid_data)
        mlflow.log_param("train_data_sha256", sha256_file(train_path))
        mlflow.log_param("valid_data_sha256", sha256_file(valid_path))
        mlflow.log_param("train_row_count", len(train_df))
        mlflow.log_param("valid_row_count", len(valid_df))

        # 3. 성능 metric 기록
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("r2", metrics["r2"])

        # 4. artifact 기록
        mlflow.log_artifact(str(feature_path))
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"run_id: {run.info.run_id}")
def main():
    args = parse_args()

    train_path = resolve_data_path(args.train_data)
    valid_path = resolve_data_path(args.valid_data)

    import pandas as pd

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    x_train, y_train = split_features_target(train_df)
    x_valid, y_valid = split_features_target(valid_df)

    model = build_model(args)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_valid, y_valid)
    feature_path = write_feature_artifact()

    print("Training finished")
    print(f"run_name: {args.run_name}")
    print(f"model_type: {args.model_type}")
    print(f"train_data: {args.train_data}")
    print(f"train_rows: {len(train_df)}")
    print(f"train_data_sha256: {sha256_file(train_path)}")
    print(f"rmse: {metrics['rmse']:.4f}")
    print(f"mae: {metrics['mae']:.4f}")
    print(f"r2: {metrics['r2']:.4f}")

    log_experiment_to_mlflow(args, model, metrics, feature_path, train_df, valid_df)


if __name__ == "__main__":
    main()
