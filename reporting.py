import json
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay

from common.data import prepare_for_inference
from common.model import inference, make_confusion_matrix


def score_model(
    model_path: Path,
    encoder_path: Path,
    test_data_dir: Path,
    categorical_features: list,
    Y_label: str,
    output_path: Path,
):
    model, X_test, Y_test = prepare_for_inference(
        test_data_dir, model_path, encoder_path, categorical_features, Y_label
    )
    y_pred = inference(model, X_test)
    print("Calculate confusion matrix")
    conf_matrix = make_confusion_matrix(Y_test, y_pred)
    conf_plot = output_path / "confusionmatrix.png"
    print(f"Plot confusion matrix and save in {conf_plot}")
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_)
    display.plot()
    display.figure_.savefig(conf_plot, bbox_inches="tight")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_path = Path(config["output_model_path"])
    model_path = output_path / "trainedmodel.pkl"
    encoder_path = output_path / "encoder.pkl"
    test_data_dir = Path(config["test_data_path"])
    categorical_features = config["categorical_features"]
    Y_label = config["Y_label"]

    score_model(
        model_path,
        encoder_path,
        test_data_dir,
        categorical_features,
        Y_label,
        output_path,
    )
