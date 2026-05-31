from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

FEATURE_COLUMNS = ["trip_distance", "passenger_count", "pickup_hour", "is_weekend"]
TARGET_COLUMN = "fare_amount"


def load_dataset(filename: str) -> pd.DataFrame:
    """data 디렉터리에서 csv 파일을 읽어옵니다."""
    return pd.read_csv(DATA_DIR / filename)


def split_features_target(df: pd.DataFrame):
    """학습/검증 데이터프레임을 feature와 target으로 분리합니다."""
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return x, y


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """학습과 서빙에서 동일한 feature 순서를 사용하도록 고정합니다."""
    return df[FEATURE_COLUMNS]


def request_to_dataframe(features: dict) -> pd.DataFrame:
    """API 요청으로 들어온 feature dict를 모델 입력용 DataFrame으로 변환합니다."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in features]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    return pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
