import json
from pathlib import Path

from flask import Flask, request

from common.data import merge_multiple_dataframe, prepare_for_inference
from common.model import inference
from diagnostics import (
    dataframe_summary,
    missing_data,
    execution_time,
    outdated_packages_list,
)
from scoring import score_model


app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_dir = Path(config["output_folder_path"])
model_dir = Path(config["output_model_path"])
categorical_features = config["categorical_features"]
Y_label = config["Y_label"]

prediction_model = None


@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    model, X_test, Y_test = prepare_for_inference(
        Path(request.args.get("data")),
        model_dir / "trainedmodel.pkl",
        model_dir / "encoder.pkl",
        categorical_features,
        Y_label,
    )
    prediction = [
        x.item() for x in inference(model, X_test)
    ]  # flask doesn't support numpy types
    return prediction


@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    return [
        score_model(
            model_dir / "trainedmodel.pkl",
            model_dir / "encoder.pkl",
            dataset_csv_dir,
            categorical_features,
            Y_label,
        )
    ]


@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    df = merge_multiple_dataframe(dataset_csv_dir)
    numerical_data = df.drop(*[categorical_features], axis=1)
    return [dataframe_summary(numerical_data)]


@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnose():
    df = merge_multiple_dataframe(dataset_csv_dir)
    miss = missing_data(df)

    timing = execution_time()

    dep = outdated_packages_list()
    return [miss, timing, dep]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
