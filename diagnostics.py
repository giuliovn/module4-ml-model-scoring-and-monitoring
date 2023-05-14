import json
from pathlib import Path

import joblib
import pandas as pd

from ingestion import merge_multiple_dataframe
from common.model import inference


def model_predictions(model_path: Path, data: pd.DataFrame):
    print(f"Load model {model_path}")
    model = joblib.load(model_path)
    return inference(model, data)


# ##################Function to get summary statistics
# def dataframe_summary():
#     #calculate summary statistics here
#     return #return value should be a list containing all summary statistics
#
# ##################Function to get timings
# def execution_time():
#     #calculate timing of training.py and ingestion.py
#     return #return a list of 2 timing values in seconds
#
# ##################Function to check dependencies
# def outdated_packages_list():
#     #get a list of


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    output_path = Path(config["output_folder_path"])
    test_data_dir = Path(config["output_folder_path"])
    # categorical_features = config["categorical_features"]
    # Y_label = config["Y_label"]
    print(f"Read data in {test_data_dir}")
    df = merge_multiple_dataframe(test_data_dir)

    model_predictions(output_path / "trainedmodel.pkl", df)
    # dataframe_summary()
    # execution_time()
    # outdated_packages_list()
