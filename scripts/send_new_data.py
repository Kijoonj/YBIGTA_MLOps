import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solution.src.features import FEATURE_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description="Send a CSV dataset to the production prediction API.")
    parser.add_argument("--data", default="data/new_data.csv")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def resolve_root_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def post_prediction(url, features):
    payload = json.dumps({"features": features}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    args = parse_args()
    data_path = resolve_root_path(args.data)
    df = pd.read_csv(data_path)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    responses = []
    for _, row in df.iterrows():
        features = {column: float(row[column]) for column in FEATURE_COLUMNS}
        responses.append(post_prediction(args.url, features))

    report_path = responses[-1].get("drift_report_path") if responses else None
    print(f"sent_rows: {len(responses)}")
    if responses:
        print(f"last_prediction: {responses[-1]['prediction']}")
        print(f"model_run_id: {responses[-1]['model_run_id']}")
    if report_path:
        print(f"drift_report_path: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        print(f"Failed to call prediction API: {exc}", file=sys.stderr)
        sys.exit(1)
