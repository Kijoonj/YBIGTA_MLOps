import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solution.src.features import FEATURE_COLUMNS, load_dataset


REPORT_PATH = ROOT_DIR / "solution" / "reports" / "drift_report.txt"
DRIFT_THRESHOLD = 2.0


def main():
    train_df = load_dataset("train.csv")
    new_df = load_dataset("new_data.csv")

    feature_reports = []
    drift_detected = False

    for feature in FEATURE_COLUMNS:
        train_mean = train_df[feature].mean()
        new_mean = new_df[feature].mean()
        train_std = train_df[feature].std()

        if train_std == 0:
            drift_score = 0.0
        else:
            drift_score = abs(new_mean - train_mean) / train_std

        status = "drift_detected" if drift_score >= DRIFT_THRESHOLD else "ok"
        if status == "drift_detected":
            drift_detected = True

        feature_reports.append(
            "\n".join(
                [
                    f"feature: {feature}",
                    f"train_mean: {train_mean:.4f}",
                    f"new_mean: {new_mean:.4f}",
                    f"train_std: {train_std:.4f}",
                    f"drift_score: {drift_score:.4f}",
                    f"status: {status}",
                ]
            )
        )

    overall_status = "WARNING" if drift_detected else "OK"
    report_text = f"DRIFT_CHECK: {overall_status}\n\n" + "\n\n".join(feature_reports)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"DRIFT_CHECK: {overall_status}")
    print(f"drift_report_path: {REPORT_PATH}")


if __name__ == "__main__":
    main()
