import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def pass_msg(message):
    print(f"[PASS] {message}")


def fail_msg(message):
    print(f"[FAIL] {message}")


def check(condition, message, failures):
    if condition:
        pass_msg(message)
    else:
        fail_msg(message)
        failures.append(message)


def get_mlflow_run_dirs():
    mlruns_dir = ROOT_DIR / "mlruns"
    if not mlruns_dir.exists():
        return []

    run_dirs = []
    for meta_path in mlruns_dir.rglob("meta.yaml"):
        run_dir = meta_path.parent
        if (run_dir / "params").exists() or (run_dir / "metrics").exists():
            run_dirs.append(run_dir)
    return run_dirs


def has_param(run_dir, param_name):
    return (run_dir / "params" / param_name).exists()


def count_runs_with_param(param_name):
    return sum(1 for run_dir in get_mlflow_run_dirs() if has_param(run_dir, param_name))


def read_param_values(param_name):
    values = set()
    for run_dir in get_mlflow_run_dirs():
        param_path = run_dir / "params" / param_name
        if param_path.exists():
            values.add(param_path.read_text(encoding="utf-8").strip())
    return values


def check_health_endpoint():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
            if response.status != 200:
                return False, f"status code {response.status}"
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("model_loaded") is True, str(payload)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, str(exc)


def read_csv_header(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            return next(reader)
        except StopIteration:
            return []


def main():
    failures = []

    mlruns_dir = ROOT_DIR / "mlruns"
    production_model_dir = ROOT_DIR / "solution" / "models" / "production_model"
    metadata_path = production_model_dir / "metadata.json"
    prediction_log_path = ROOT_DIR / "solution" / "logs" / "predictions.csv"
    drift_report_path = ROOT_DIR / "solution" / "reports" / "drift_report.txt"

    run_dirs = get_mlflow_run_dirs()
    train_data_values = read_param_values("train_data_path")
    model_type_values = read_param_values("model_type")

    check(mlruns_dir.exists(), "mlruns directory exists", failures)
    check(len(run_dirs) >= 4, "at least 4 MLflow runs found", failures)
    check(len(train_data_values) >= 3, "at least 3 different training datasets logged", failures)
    check(len(model_type_values) >= 2, "at least 2 different model types logged", failures)
    check(count_runs_with_param("train_data_sha256") >= 4, "train data sha256 logged for each experiment", failures)
    check(count_runs_with_param("valid_data_sha256") >= 4, "valid data sha256 logged for each experiment", failures)
    check(count_runs_with_param("train_row_count") >= 4, "training row count logged for each experiment", failures)

    check(production_model_dir.exists(), "production model directory exists", failures)
    check(metadata_path.exists(), "production metadata exists", failures)

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            required_keys = {
                "run_id",
                "model_type",
                "train_data_path",
                "train_data_sha256",
                "metric_name",
                "metric_value",
                "promoted_at",
            }
            check(required_keys.issubset(metadata.keys()), "production metadata has required keys", failures)
            required_values = {key: metadata.get(key) for key in required_keys}
            check(all(value not in (None, "") for value in required_values.values()), "production metadata values are not empty", failures)
        except json.JSONDecodeError:
            check(False, "production metadata is valid json", failures)

    health_ok, health_detail = check_health_endpoint()
    check(health_ok, f"FastAPI health check returns 200 and model_loaded=true ({health_detail})", failures)

    check(prediction_log_path.exists(), "prediction log exists", failures)
    if prediction_log_path.exists():
        header = read_csv_header(prediction_log_path)
        check({"timestamp", "model_run_id", "prediction"}.issubset(header), "prediction log has required columns", failures)
        required_features = {"trip_distance", "passenger_count", "pickup_hour", "is_weekend"}
        check(required_features.issubset(header), "prediction log includes feature columns", failures)

    check(drift_report_path.exists(), "drift report exists", failures)
    if drift_report_path.exists():
        report_text = drift_report_path.read_text(encoding="utf-8")
        check("DRIFT_CHECK:" in report_text, "drift status calculated", failures)
        check("drift_score" in report_text, "drift score included in report", failures)
        check("reference_train_data_path" in report_text, "drift report includes reference training data", failures)
        check("request_count" in report_text, "drift report includes request count", failures)

    print()
    if failures:
        print("RESULT: FAIL")
        print("다음 항목을 다시 확인하세요:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("RESULT: PASS")


if __name__ == "__main__":
    main()
