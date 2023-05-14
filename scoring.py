import json
from pathlib import Path

import joblib

from common.files import save_file
from common.data import process_data
from common.model import evaluate_regression_model
from ingestion import merge_multiple_dataframe


def score_model(
    model_path: Path,
    encoder_path: Path,
    test_data_dir: Path,
    categorical_features: list,
    Y_label: str,
):
    print(f"Read data in {test_data_dir}")
    df = merge_multiple_dataframe(test_data_dir)
    print(f"Load model {model_path}")
    model = joblib.load(model_path)
    print(f"Load encoder {encoder_path}")
    encoder = joblib.load(encoder_path)
    print("Process data")
    X_test, Y_test, _ = process_data(
        df,
        categorical_features=categorical_features,
        label=Y_label,
        encoder=encoder,
        training=False,
    )
    print("Evaluate")
    precision, recall, fbeta = evaluate_regression_model(model, X_test, Y_test)
    print(f"Precision: {precision}. Recall: {recall}. Fbeta: {fbeta}")
    return fbeta


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_path = Path(config["output_model_path"])
    model_path = output_path / "trainedmodel.pkl"
    encoder_path = output_path / "encoder.pkl"
    test_data_dir = Path(config["test_data_path"])
    categorical_features = config["categorical_features"]
    Y_label = config["Y_label"]

    fbeta = score_model(
        model_path, encoder_path, test_data_dir, categorical_features, Y_label
    )
    latest_score = output_path / "latestscore.txt"
    print(f"Save F1 score to {latest_score}")
    save_file(str(fbeta), latest_score, format="txt")
