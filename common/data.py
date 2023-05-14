from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from common.files import save_file


def merge_multiple_dataframe(input_folder_path: Path, output_folder_path: Path = None):
    # check for datasets, compile them together, and write to an output Path
    print(f"Find and concatenate csvs in {input_folder_path}")
    df = pd.DataFrame()
    ingest_files = []
    for csv_file in input_folder_path.glob("*.csv"):
        ingest_files.append(str(csv_file))
        df = pd.concat([df, pd.read_csv(csv_file)], ignore_index=True)

    print("Deduplicate results")
    df = df.drop_duplicates()

    if output_folder_path:
        final_csv = output_folder_path / "finaldata.csv"
        print(f"Save to {final_csv}")
        save_file(df, final_csv, format="csv")

        ingest_data_txt = output_folder_path / "ingestedfiles.txt"
        save_file(ingest_files, ingest_data_txt, format="txt")

    return df


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


def prepare_for_inference(
    data_path: Path,
    model_path: Path,
    encoder_path: Path,
    categorical_features: list,
    Y_label: str,
):
    if data_path.is_dir():
        print(f"Read data in {data_path} directory")
        df = merge_multiple_dataframe(data_path)
    if data_path.is_file():
        print(f"Read data in {data_path} file")
        df = pd.read_csv(data_path)

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
    return model, X_test, Y_test
