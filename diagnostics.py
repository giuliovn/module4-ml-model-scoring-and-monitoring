import json
from pathlib import Path
import subprocess
import timeit

import pandas as pd

from ingestion import merge_multiple_dataframe
from common.data import prepare_for_inference


def dataframe_summary(numerical_data: pd.DataFrame):
    print("Calculate mean, median and standard deviation on numerical features")
    mean = list(numerical_data.mean())
    median = list(numerical_data.median())
    std = list(numerical_data.std())
    return mean, median, std


def missing_data(data: pd.DataFrame):
    print("Calculate percent of missing data")
    nas = list(data.isna().sum())
    napercents = [nas[i] / len(data.index) for i in range(len(nas))]
    print(napercents)
    return napercents


def execution_time():
    print("Calculate ingestion timing")
    start_ingestion = timeit.default_timer()
    subprocess.check_output(["python3", "ingestion.py"])
    ingestion_timing = timeit.default_timer() - start_ingestion

    print("Calculate train timing")
    start_train = timeit.default_timer()
    subprocess.check_output(["python3", "training.py"])
    train_timing = timeit.default_timer() - start_train
    timing = [ingestion_timing, train_timing]
    print(timing)
    return timing


def outdated_packages_list():
    print("Check dependencies")
    stdout, _ = subprocess.Popen(
        ["pip", "list", "--outdated"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()
    dep = stdout.decode("utf-8").strip("\n")
    print(dep)
    return dep


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_folder_path = Path(config["output_folder_path"])
    output_model_path = Path(config["output_model_path"])
    categorical_features = config["categorical_features"]
    Y_label = config["Y_label"]
    print(f"Read data in {output_folder_path}")
    df = merge_multiple_dataframe(output_folder_path)

    model, X_test, Y_test = prepare_for_inference(
        output_folder_path,
        output_model_path / "trainedmodel.pkl",
        output_model_path / "encoder.pkl",
        categorical_features,
        Y_label,
    )

    numerical_data = df.drop(*[categorical_features], axis=1)
    dataframe_summary(numerical_data)

    missing_data(df)

    execution_time()

    outdated_packages_list()
