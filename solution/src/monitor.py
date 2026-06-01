import sys
import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solution.src.features import FEATURE_COLUMNS


PRODUCTION_MODEL_DIR = ROOT_DIR / "solution" / "models" / "production_model"
METADATA_PATH = PRODUCTION_MODEL_DIR / "metadata.json"
PREDICTION_LOG_PATH = ROOT_DIR / "solution" / "logs" / "predictions.csv"
REPORT_PATH = ROOT_DIR / "solution" / "reports" / "drift_report.txt"
DRIFT_THRESHOLD = 2.0


def resolve_root_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Production metadata not found: {METADATA_PATH}")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_reference_data(metadata):
    train_data_path = metadata.get("train_data_path") or "data/train.csv"
    resolved_path = resolve_root_path(train_data_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Reference training data not found: {resolved_path}")
    return pd.read_csv(resolved_path), train_data_path


def load_prediction_requests(metadata):
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(f"Prediction log not found: {PREDICTION_LOG_PATH}")

    request_df = pd.read_csv(PREDICTION_LOG_PATH)
    run_id = metadata.get("run_id")
    if run_id and "model_run_id" in request_df.columns:
        request_df = request_df[request_df["model_run_id"] == run_id]

    if request_df.empty:
        raise ValueError("No prediction requests found for the current production model.")

    return request_df


def build_drift_report(train_df, request_df, metadata, reference_path):
    missing_columns = [column for column in FEATURE_COLUMNS if column not in request_df.columns]
    if missing_columns:
        raise ValueError(f"Prediction log is missing feature columns: {missing_columns}")

    feature_reports = []
    drift_detected = False

    for feature in FEATURE_COLUMNS:
        # TODO: reference training data와 prediction request data의 feature 통계를 계산하세요.
        train_mean = ____
        request_mean = ____
        train_std = ____

        if pd.isna(train_std) or train_std == 0:
            drift_score = 0.0
        else:
            drift_score = abs(request_mean - train_mean) / train_std

        status = "drift_detected" if drift_score >= DRIFT_THRESHOLD else "ok"
        if status == "drift_detected":
            drift_detected = True

        feature_reports.append(
            "\n".join(
                [
                    f"feature: {feature}",
                    f"train_mean: {train_mean:.4f}",
                    f"request_mean: {request_mean:.4f}",
                    f"train_std: {train_std:.4f}",
                    f"drift_score: {drift_score:.4f}",
                    f"status: {status}",
                ]
            )
        )

    overall_status = "WARNING" if drift_detected else "OK"
    header = "\n".join(
        [
            f"DRIFT_CHECK: {overall_status}",
            f"model_run_id: {metadata.get('run_id')}",
            f"reference_train_data_path: {reference_path}",
            f"prediction_log_path: {PREDICTION_LOG_PATH}",
            f"request_count: {len(request_df)}",
        ]
    )
    return header + "\n\n" + "\n\n".join(feature_reports)


def write_drift_report():
    metadata = load_metadata()
    train_df, reference_path = load_reference_data(metadata)
    request_df = load_prediction_requests(metadata)
    report_text = build_drift_report(train_df, request_df, metadata, reference_path)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    return REPORT_PATH


def main():
    report_path = write_drift_report()
    report_text = report_path.read_text(encoding="utf-8")
    first_line = report_text.splitlines()[0]

    print(first_line)
    print(f"drift_report_path: {report_path}")


if __name__ == "__main__":
    main()
