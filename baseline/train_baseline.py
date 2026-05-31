from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "baseline" / "models"

FEATURE_COLUMNS = ["trip_distance", "passenger_count", "pickup_hour", "is_weekend"]
TARGET_COLUMN = "fare_amount"


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid.csv")

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET_COLUMN]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_valid)
    rmse = mean_squared_error(y_valid, predictions) ** 0.5
    mae = mean_absolute_error(y_valid, predictions)
    r2 = r2_score(y_valid, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "baseline_model.joblib"
    joblib.dump(model, model_path)

    print("Baseline training finished")
    print(f"model_path: {model_path}")
    print(f"rmse: {rmse:.4f}")
    print(f"mae: {mae:.4f}")
    print(f"r2: {r2:.4f}")
    print()
    print("운영 관점에서 확인할 점:")
    print("- 실험 run id가 없어 여러 실험을 비교하기 어렵습니다.")
    print("- 모델 metadata가 없어 어떤 파라미터로 학습했는지 추적하기 어렵습니다.")
    print("- production model과 baseline model의 구분이 없습니다.")


if __name__ == "__main__":
    main()
