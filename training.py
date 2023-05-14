import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from common.files import save_file
from common.data import process_data
from common.model import evaluate_regression_model


def train_model(
    input_file: Path, categorical_features: list, Y_label: str, output_model_path: Path
):
    print(f"Read {input_file}")
    df = pd.read_csv(input_file)

    print("Train")
    # use this logistic regression for training
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    train, test = train_test_split(df, test_size=0.20)

    X_train, Y_train, encoder = process_data(
        train, categorical_features=categorical_features, label=Y_label, training=True
    )
    X_test, Y_test, _ = process_data(
        test,
        categorical_features=categorical_features,
        label=Y_label,
        encoder=encoder,
        training=False,
    )
    lr.fit(X_train, Y_train)

    print("Evaluate")
    precision, recall, fbeta = evaluate_regression_model(lr, X_test, Y_test)
    print(f"Precision: {precision}. Recall: {recall}. Fbeta: {fbeta}")

    # write the trained model to your workspace in a file called trainedmodel.pkl
    model_pkl = output_model_path / "trainedmodel.pkl"
    print(f"Save model to {model_pkl}")
    save_file(lr, model_pkl, format="pkl")
    encoder_pkl = output_model_path / "encoder.pkl"
    print(f"Save encoder to {encoder_pkl}")
    save_file(encoder, encoder_pkl, format="pkl")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_folder_path = Path(config["output_folder_path"])
    output_model_path = Path(config["output_model_path"])
    input_file = output_folder_path / "finaldata.csv"
    categorical_features = config["categorical_features"]
    Y_label = config["Y_label"]

    train_model(input_file, categorical_features, Y_label, output_model_path)
