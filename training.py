import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train_model(input_file):
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
    categorical_features = ["corporation"]
    X_train, Y_train, encoder = process_data(
        train, categorical_features=categorical_features, label="exited", training=True
    )
    X_test, Y_test, _ = process_data(
        test,
        categorical_features=categorical_features,
        label="exited",
        encoder=encoder,
        training=False,
    )
    lr.fit(X_train, Y_train)

    print("Evaluate")
    y_pred = lr.predict(X_test)
    precision = precision_score(Y_test, y_pred, zero_division=1)
    recall = recall_score(Y_test, y_pred, zero_division=1)
    fbeta = fbeta_score(Y_test, y_pred, beta=1, zero_division=1)
    print(f"Precision: {precision}. Recall: {recall}. Fbeta: {fbeta}")

    # write the trained model to your workspace in a file called trainedmodel.pkl
    model_pkl = output_folder_path / "trainedmodel.pkl"
    print(f"Save model to {model_pkl}")
    joblib.dump(lr, model_pkl)


def process_data(X, categorical_features=[], label=None, training=True, encoder=None):
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
    else:
        X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_folder_path = Path(config["output_folder_path"])
    input_file = output_folder_path / "finaldata.csv"

    train_model(input_file)
