import json
from pathlib import Path

from common.files import save_file
from common.data import prepare_for_inference
from common.model import evaluate_regression_model, inference


def score_model(
    model_path: Path,
    encoder_path: Path,
    test_data_dir: Path,
    categorical_features: list,
    Y_label: str,
):
    model, X_test, Y_test = prepare_for_inference(
        test_data_dir, model_path, encoder_path, categorical_features, Y_label
    )
    y_pred = inference(model, X_test)
    print("Evaluate")
    precision, recall, fbeta = evaluate_regression_model(Y_test, y_pred)
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
